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
Provides support to run compiled networks both locally and remotely.
"""
import json
import logging
import os
import tarfile
import tempfile

import numpy as np
import tvm
from tvm import rpc
from tvm.autotvm.measure import request_remote
from tvm.contrib import graph_runtime as runtime
from tvm.contrib.debugger import debug_runtime

from tvmc.common import TVMCException


def get_input_info(graph_str, params):
    """Return the shape and dtype dictionaries for the input
    tensors of a compiled module.

    Parameters
    ----------
    graph_str : str
        JSON graph of the module serialised as a string.
    params : bytearray
        Params serialised as a bytearray.

    Returns
    -------
    shape_dict : dict
        Shape dictionary - {input_name: tuple}.
    dtype_dict : dict
        dtype dictionary - {input_name: dtype}.

    """
    # NOTE - We can't simply get the input tensors from a TVM graph
    # because weight tensors are treated equivalently. Therefore, to
    # find the input tensors we look at the 'arg_nodes' in the graph
    # (which are either weights or inputs) and check which ones don't
    # appear in the params (where the weights are stored). These nodes
    # are therefore inferred to be input tensors.

    shape_dict = {}
    dtype_dict = {}
    # Use a special function to load the binary params back into a dict
    load_arr = tvm.get_global_func("tvm.relay._load_param_dict")(params)
    param_names = [v.name for v in load_arr]
    graph = json.loads(graph_str)
    for node_id in graph["arg_nodes"]:
        node = graph["nodes"][node_id]
        # If a node is not in the params, infer it to be an input node
        name = node["name"]
        if name not in param_names:
            shape_dict[name] = graph["attrs"]["shape"][1][node_id]
            dtype_dict[name] = graph["attrs"]["dltype"][1][node_id]

    return shape_dict, dtype_dict


def generate_tensor_data(shape, dtype, fill_mode):
    """Generate data to produce a tensor of given shape and dtype.

    Random data generation depends on the dtype. For int8 types,
    random integers in the range 0->255 are generated. For all other
    types, random floats are generated in the range -1->1 and then
    cast to the appropriate dtype.

    Parameters
    ----------
    shape : tuple
        The shape of the tensor.
    dtype : str
        The dtype of the tensor.
    fill_mode : str
        The fill-mode to use, either "zeros", "ones" or "random".

    Returns
    -------
    tensor : np.array
        The generated tensor as a np.array.

    """
    if fill_mode == "zeros":
        return np.zeros(shape=shape, dtype=dtype)

    if fill_mode == "ones":
        return np.ones(shape=shape, dtype=dtype)

    if fill_mode == "random":
        np.random.seed(42)
        if "int8" in dtype:
            return np.random.randint(256, size=shape, dtype=dtype)

        return np.random.uniform(-1, 1, size=shape).astype(dtype)

    raise TVMCException("unknown fill-mode: {}".format(fill_mode))


def make_inputs_dict(inputs, shape_dict, dtype_dict, fill_mode):
    """Make the inputs dictionary for a graph.

    Use data from 'inputs' where specified. For input tensors
    where no data has been given, generate data according to the
    chosen fill-mode.

    Parameters
    ----------
    inputs : dict
        Input data dictionary - {input_name: np.array}.
    shape_dict : dict
        Shape dictionary - {input_name: tuple}.
    dtype_dict : dict
        dtype dictionary - {input_name: dtype}.
    fill_mode : str
        The fill-mode to use when generating tensor data.
        Can be either "zeros", "ones" or "random".

    Returns
    -------
    inputs_dict : dict
        Complete inputs dictionary - {input_name: np.array}.

    """
    # First check all the keys in inputs exist in the graph
    for input_name in inputs:
        if input_name not in shape_dict.keys():
            raise TVMCException(
                "the input tensor '{}' is not in the graph".format(input_name)
            )

    # Now construct the input dict, generating tensors where no
    # data already exists in 'inputs'
    inputs_dict = {}
    for input_name in shape_dict:
        if input_name in inputs.keys():
            inputs_dict[input_name] = inputs[input_name]
        else:
            shape = shape_dict[input_name]
            dtype = dtype_dict[input_name]
            data = generate_tensor_data(shape, dtype, fill_mode)
            inputs_dict[input_name] = data

    return inputs_dict


def run_module(
        module_file,
        hostname,
        port=9090,
        tracker_key=None,
        inputs=None,
        fill_mode="zeros",
        repeat=1,
        profile=False,
):
    """Run a compiled graph runtime module locally or remotely with
    optional input values.

    If input tensors are not specified explicitly, they can be filled
    with zeroes, ones or random data.

    Parameters
    ----------
    module_file : str
        The path to the module file (a .tar file).
    hostname : str
        The hostname of the target device on which to run.
    port : int
        The port of the target device on which to run.
    tracker_key : str, optional
        The tracker key of the target device. If this is set, it
        will be assumed that remote points to a tracker.
    inputs : dict, optional
        A dictionary of {input_name: np.array} storing the input
        tensors to the network. If this is not specified, data will
        be automatically generated for the input tensors.
    fill_mode : str, optional
        The fill-mode to use when generating data for input tensors.
        Valid options are "zeros", "ones" and "random".
        Defaults to "zeros".
    repeat : int, optional
        How many times to repeat the run.
    profile : bool
        Whether to profile the run with the debug runtime.

    Returns
    -------

    """
    if not inputs:
        inputs = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        # First untar and load the module file
        logging.debug("extracting module file")
        t = tarfile.open(module_file)
        t.extractall(tmp_dir)
        graph = open(os.path.join(tmp_dir, "mod.json")).read()
        params = bytearray(open(os.path.join(tmp_dir, "mod.params"), "rb").read())

        # Initiate RPC connection
        if hostname:
            # If a tracker_key is set, then the remote is a tracker.
            if tracker_key:
                remote_device = request_remote(
                    tracker_key, hostname, port, timeout=1000
                )
            else:
                # Otherwise it's a RPCServer hosted on a device.
                remote_device = rpc.connect(hostname, port)
        else:
            remote_device = rpc.LocalSession()

        remote_device.upload(os.path.join(tmp_dir, "mod.so"))
        remote_lib = remote_device.load_module("mod.so")

        # TODO Support non-cpu targets
        ctx = remote_device.cpu(0)
        logging.debug("Create the runtime")
        if profile:
            module = debug_runtime.create(graph, remote_lib, ctx, dump_root="./prof")
        else:
            module = runtime.create(graph, remote_lib, ctx)

        logging.debug("Load the params (weights) into the runtime module")
        module.load_params(params)

        # Load the inputs into the runtime module
        shape_dict, dtype_dict = get_input_info(graph, params)
        inputs_dict = make_inputs_dict(inputs, shape_dict, dtype_dict, fill_mode)
        logging.debug("Setting inputs")
        module.set_input(**inputs_dict)

        logging.debug("Running the module")
        if profile:
            logging.debug("Profiling is enabled")
            module.run()

        timer = module.module.time_evaluator("run", ctx, 1, repeat=repeat)
        prof_result = timer()
        times = prof_result.results

        # Get the output tensors
        num_outputs = module.get_num_outputs()
        outputs = {}
        logging.debug("Outputs: %s", num_outputs)
        for i in range(num_outputs):
            output_name = "output_{}".format(i)
            outputs[output_name] = module.get_output(i).asnumpy()

        return outputs, times


def get_top_results(outputs, max_results):
    """Return the top n results from the output tensor.

    This function is primarily for image classification and will
    not necessarily generalize.

    Parameters
    ----------
    outputs : dict
        Outputs dictionary - {output_name: np.array}.
    max_results : int
        Number of results to return

    Returns
    -------
    top_results : np.array
        Results array of shape (2, n).
        The first row is the indices and the second is the values.

    """
    output = outputs["output_0"]
    sorted_labels = output.argsort()[0][-max_results:][::-1]
    output.sort()
    sorted_values = output[0][-max_results:][::-1]
    top_results = np.array([sorted_labels, sorted_values])
    return top_results


def format_times(times):
    """Format the mean, max, min and std of the times.

    Parameters
    ----------
    times : list
        A list of the times (in seconds).

    Returns
    -------
    stat_table : str
        A formatted string containing the statistics.
    """
    # This has the effect of producing a small table that looks like:
    # mean (s)   max (s)    min (s)    std (s)
    # 0.14310    0.16161    0.12933    0.01004
    mean_ts = np.mean(times)
    std_ts = np.std(times)
    max_ts = np.max(times)
    min_ts = np.min(times)
    header = "{0:^10} {1:^10} {2:^10} {3:^10}".format(
        "mean (s)", "max (s)", "min (s)", "std (s)"
    )
    stats = "{0:^10.5f} {1:^10.5f} {2:^10.5f} {3:^10.5f}".format(mean_ts, max_ts, min_ts, std_ts)
    return header + "\n" + stats
