# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Provides support to compile networks both AOT and JIT.
"""
import logging
import tarfile
from pathlib import Path

import tvm
from tvm import autotvm
from tvm import relay
from tvm._ffi.runtime_ctypes import TVMContext
from tvm.contrib import cc
from tvm.contrib import util
from tvm.relay.op.contrib import get_pattern_table

from tvmc import frontends
from tvmc.common import convert_graph_layout


def compile_model(
        path,
        targets,
        dump_sources=None,
        target_host=None,
        language=None,
        sanitize_diagnostics=True,
        shapes=None,
        tuner_file=None,
        alter_layout=None,
):
    """Compile a model from a supported framework into a TVM module.

    This function takes a union of the arguments of both frontends.load_model
    and compiler.compile_relay. The resulting TVM module can be executed using
    the graph runtime.

    Returns
    -------
    graph : str
        A JSON-serialized TVM execution graph.
    lib : tvm.module.Module
        A TVM module containing the compiled functions.
    params : dict
        The parameters (weights) for the TVM module.
    dumps : dict
            Dictionary containing the dumps specified.

    """
    mod, params = frontends.load_model(path, language, sanitize_diagnostics, shapes)
    return compile_relay(
        mod,
        params,
        targets,
        dump_sources=dump_sources,
        target_host=target_host,
        tuner_file=tuner_file,
        alter_layout=alter_layout,
    )


def compile_relay(
        mod,
        params,
        targets,
        dump_sources=None,
        target_host=None,
        tuner_file=None,
        alter_layout=None,
):
    """Compile a relay module to a TVM module for the graph runtime.

    Parameters
    ----------
    mod : tvm.relay.Module
        The relay module to compile.
    params : dict
        The parameters (weights) for the relay module.
    targets : list(Union[str, tvm.target.Target])
        The targets for which to compile in priority order.
    dump_sources : list, optional
        Dump the generated code for the specified source types.
        "ll" - LLVM IR
        "asm" - Assembly
    target_host : Union[str, tvm.target.Target], optional
        The target of the host machine if host-side code
        needs to be generated.
    tuner_file: str, optional
        Name of the file produced by the tuning to be used during
        compilation.
    alter_layout: str, optional
        The layout to convert the graph to. Note, the convert layout
        pass doesn't currently guarantee the whole of the graph will
        be converted to the chosen layout.

    Returns
    -------
    graph : str
        A JSON-serialized TVM execution graph.
    lib : tvm.module.Module
        A TVM module containing the compiled functions.
    params : dict
        The parameters (weights) for the TVM module.
    dumps : dict
        Dictionary containing the dumps specified.

    """
    source_types = ["relay", "ll", "asm"]
    if not dump_sources:
        dump_sources = []

    tvm_targets, external_compilers, cpu_target = create_targets(targets)

    # if the target host isn't specified, default to using the cpu target
    if not target_host:
        target_host = cpu_target

    if alter_layout:
        mod = convert_graph_layout(mod, alter_layout)

    for external_compiler in external_compilers:
        logging.debug("external partitioning - %s", external_compiler)
        f = relay.build_module.bind_params_by_name(mod["main"], params)
        mod = tvm.IRModule()
        mod["main"] = f

        pattern_table = get_pattern_table(external_compiler)
        mod = relay.transform.MergeComposite(pattern_table)(mod)
        mod = relay.transform.AnnotateTarget(external_compiler)(mod)
        mod = relay.transform.MergeCompilerRegions()(mod)
        mod = relay.transform.PartitionGraph()(mod)

    dumps = {}

    if tuner_file:
        logging.debug("a tuner file is provided %s", tuner_file)
        with autotvm.apply_history_best(tuner_file):
            with tvm.transform.PassContext(opt_level=3):
                logging.debug("building relay graph with tuner file")
                graph_mod = relay.build(
                    mod, cpu_target, params=params, target_host=target_host
                )
    else:
        with tvm.transform.PassContext(opt_level=3):
            logging.debug("building relay graph without tuner file")
            graph_mod = relay.build(
                mod, tvm_targets, params=params, target_host=target_host
            )

    for source_type in dump_sources:
        logging.debug("generating dump file %s", source_type)
        if source_type in source_types:
            lib = graph_mod.get_lib()
            source = str(mod) if source_type == "relay" else lib.get_source(source_type)
            dumps[source_type] = source

    return graph_mod.get_json(), graph_mod.get_lib(), graph_mod.get_params(), dumps


def save_module(module_path, graph, lib, params, cross=None):
    """
    Create a tarball containing the generated TVM graph,
    exported library and parameters

    Parameters
    ----------
    module_path : str
        path to the target tar.gz file to be created,
        including the file name
    graph : str
        A JSON-serialized TVM execution graph.
    lib : tvm.module.Module
        A TVM module containing the compiled functions.
    params : dict
        The parameters (weights) for the TVM module.
    cross : Union[str, Callable[[str, str, Optional[str]], None]]
        Function that performs the actual compilation

    """
    lib_name = "mod.so"
    graph_name = "mod.json"
    param_name = "mod.params"
    temp = util.tempdir()
    path_lib = temp.relpath(lib_name)
    if not cross:
        lib.export_library(path_lib)
    else:
        lib.export_library(path_lib, cc.cross_compiler(cross))

    with open(temp.relpath(graph_name), "w") as graph_file:
        graph_file.write(graph)

    with open(temp.relpath(param_name), "wb") as params_file:
        params_file.write(relay.save_param_dict(params))

    with tarfile.open(module_path, "w") as tar:
        tar.add(path_lib, lib_name)
        tar.add(temp.relpath(graph_name), graph_name)
        tar.add(temp.relpath(param_name), param_name)


def save_dumps(module_name, dumps, dump_root="."):
    """
    Serialize dump files to the disk.

    Parameters
    ----------
    module_name : list(Union[str, tvm.target.Target])
        file name, referring to the module that generated
        the dump contents
    dumps : dict
        the output contents to be saved into the files
    dump_root : str
        path in which dump files will be created
    """

    for dump_format in dumps:
        dump_name = module_name + "." + dump_format
        with open(Path(dump_root, dump_name), "w") as f:
            f.write(dumps[dump_format])


def create_targets(targets):
    """Create the targets for which to compile with.

    Currently there exists two methods of targeting a compiler: targets
    built into tvm, and external targets (BYOC). Handle both cases here.

    Parameters
    ----------
    targets : list(Union[str, tvm.target.Target])
        The targets for which to compile in priority order.

    Returns
    -------
    tvm_targets : Union[Dict[str, tvm.target.Target], tvm.target.Target]
        The targets for which to compile.
    external_compilers : List[str]
        The external compilers to use in priority order.
    cpu_target : tvm.target.Target
        The target for the cpu (commonly used as a fallback when
        other targets are used)

    """
    tvm_targets = []
    external_compilers = []
    cpu_target = "llvm"

    for target in set(targets):
        tvm_target = target
        # turn string targets into tvm.target.Targets
        if isinstance(target, str):
            tvm_target = tvm.target.create(target)

        tvm_targets.append(tvm_target)

        if tvm_target.id.device_type == TVMContext.STR2MASK["cpu"]:
            cpu_target = tvm_target

    # heterogeneous case
    if len(tvm_targets) > 1:
        target_dict = {}
        for tvm_target in tvm_targets:
            target_dict[tvm_target.device_type] = tvm_target

        tvm_targets = target_dict
    # single target case
    elif len(tvm_targets) == 1:
        tvm_targets = tvm_targets[0]
    # no target -> fall back to llvm
    elif len(tvm_targets) == 0:
        tvm_targets = tvm.target.create("llvm")

    return tvm_targets, external_compilers, cpu_target
