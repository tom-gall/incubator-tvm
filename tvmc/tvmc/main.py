#!/usr/bin/env python

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
TVMC - TVM driver command-line interface
"""
import argparse
import logging
import re
import sys

import pkg_resources

import numpy as np

import tvmc.autotuner
import tvmc.compiler
import tvmc.frontends
import tvmc.runner

from tvmc.common import TVMCException

TARGET_ALIASES = {
    "aarch64": "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon",
    "armv7-a": "llvm -device=arm_cpu -mtriple=armv7a-linux-gnueabihf -mattr=+neon,+vfp4,+thumb2"
}
VALID_TARGETS = {"aarch64", "llvm", "armv7-a"}
DEFAULT_TARGET = "llvm"


def parse_target(targets_str):
    """ Parsing function for comma separated target syntax. """
    targets = targets_str.split(",")
    for target in targets:
        if target not in VALID_TARGETS:
            raise argparse.ArgumentTypeError(f"unrecognized target: {target}")
    return targets


def parse_input_shapes(shapes_str):
    """ Parsing function for tensor shape syntax. """
    shapes = []
    # Split up string into comma seperated sections ignoring commas in ()s
    match = re.findall(r"(\(.*?\)|.+?),?", shapes_str)
    if match:
        for inp in match:
            # Test for and remove brackets
            shape = re.match(r"\((.*)\)", inp)
            if shape and shape.lastindex == 1:
                # Remove white space and extract numbers
                strshape = shape[1].replace(" ", "").split(",")
                try:
                    shapes.append([int(i) for i in strshape])
                except ValueError:
                    raise argparse.ArgumentTypeError(
                        f"expected numbers in shape '{shape[1]}'"
                    )
            else:
                raise argparse.ArgumentTypeError(
                    f"missing brackets around shape '{inp}', example '(1,2,3)'"
                )
    else:
        raise argparse.ArgumentTypeError(
            f"unrecognized shapes '{shapes_str}', example '(1,2,3),(1,4),...'"
        )
    return shapes


def add_compile_parser(subparsers):
    """ Include parser for 'compile' subcommand """

    parser = subparsers.add_parser("compile", help="compile a model")
    parser.set_defaults(func=drive_compile)
    parser.add_argument(
        "--cross-compiler",
        default="",
        help="the cross compiler to use to generate target libraries",
    )
    parser.add_argument(
        "--dump-codegen", default="", help="dump generated code [relay | ll | asm]"
    )
    parser.add_argument("--execute", action="store_true", help="run the model")
    parser.add_argument("--hostname", help="the url of the host machine")
    parser.add_argument(
        "--language",
        choices=tvmc.frontends.get_frontends(),
        help="specify input language",
    )
    parser.add_argument(
        "--input-shape",
        type=parse_input_shapes,
        metavar="INPUT_SHAPE,[INPUT_SHAPE]...",
        help="for pytorch, e.g. '(1,3,224,224)'",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="a.tar",
        help="output the compiled module to an archive",
    )
    parser.add_argument(
        "--sanitize-diagnostics",
        action="store_true",
        default=True,
        dest="sanitize_diagnostics",
        help="enable diagnostic sanitization",
    )
    parser.add_argument(
        "--no-sanitize-diagnostics",
        action="store_false",
        dest="sanitize_diagnostics",
        help="disable diagnostic sanitization",
    )
    parser.add_argument(
        "--target",
        type=parse_target,
        action="append",
        metavar="TARGET[,TARGET]...",
        help=f"compilation target(s): {', '.join(VALID_TARGETS)}, default llvm",
    )
    parser.add_argument("--tuner-file", default="", help="tuner file")
    parser.add_argument(
        "--alter-layout",
        choices=["NCHW", "NHWC"],
        default=None,
        help="change the data layout of the whole graph",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="increase verbosity"
    )
    parser.add_argument("FILE")


def add_run_parser(subparsers):
    """ Include parser for 'run' subcommand """

    parser = subparsers.add_parser("run", help="run a compiled module")
    parser.set_defaults(func=drive_run)
    parser.add_argument(
        "--fill-mode",
        choices=["zeros", "ones", "random"],
        default="zeros",
        help="fill all input tensors with values",
    )
    parser.add_argument("--hostname", help="the url of the host machine")
    parser.add_argument("-i", "--inputs", help="specify a .npz input file")
    parser.add_argument("-o", "--outputs", help="specify a .npz output file")
    parser.add_argument("--port", default=9090, type=int, help="the port to connect to")
    parser.add_argument(
        "--print-top",
        type=int,
        help="print the top n values and indices of the output tensor",
    )
    parser.add_argument(
        "--profile", action="store_true", help="profile the graph runtime"
    )
    parser.add_argument("--repeat", type=int, default=1, help="repeat the run n times")
    parser.add_argument(
        "--time", action="store_true", help="record the execution time(s)"
    )
    parser.add_argument("--tracker-key", help="the tracker key of the target device")
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="increase verbosity"
    )
    parser.add_argument("FILE")


def add_tune_parser(subparsers):
    """ Include parser for 'tune' subcommand """

    parser = subparsers.add_parser("tune", help="auto-tune a graph")
    parser.set_defaults(func=drive_tune)
    parser.add_argument(
        "--early-stopping",
        type=int,
        help="minimum number of trials before early stopping",
    )
    parser.add_argument("--hostname", help="the url of the host machine")
    parser.add_argument(
        "--language",
        choices=tvmc.frontends.get_frontends(),
        help="specify input language",
    )
    parser.add_argument(
        "--number",
        default=10,
        type=int,
        help="number of runs a single repeat is made of",
    )
    parser.add_argument(
        "--min-repeat-ms", default=1000, type=int, help="minimum time to run each trial"
    )
    parser.add_argument(
        "--tuner-file", default="history.log", help="specify an output tuner log file"
    )
    parser.add_argument(
        "--parallel",
        default=4,
        type=int,
        help="the maximum number of parallel devices to use when tuning",
    )
    parser.add_argument("--port", default=9090, type=int, help="the port to connect to")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="how many times to repeat each measurement",
    )
    parser.add_argument(
        "--target", default=DEFAULT_TARGET, help="target architecture to tune for"
    )
    parser.add_argument(
        "--timeout", default=10, help="time in seconds before timing out a config"
    )
    parser.add_argument("--tracker-key", help="the tracker key of the target device")
    parser.add_argument(
        "--trials",
        type=int,
        default=1000,
        help="the maximum number of tuning trials to perform",
    )
    parser.add_argument(
        "--tuner",
        choices=["xgb", "grid", "random"],
        default="xgb",
        help="which type of tuner to use",
    )
    parser.add_argument(
        "--alter-layout",
        choices=["NCHW", "NHWC"],
        default=None,
        help="change the data layout of the whole graph",
    )
    parser.add_argument("FILE")


def concat(targets_list):
    """
    Expand a list-of-lists given by argparse,
    to be a list of targets.
    """
    result = []
    for t in targets_list:
        result.extend(t)
    return result


def drive_compile(args):
    """ Invoke tvmc.compiler module with command line arguments """

    # argparse in python >= 3.8 supports action="extend", but for
    # versions before that, the poor mans alternative is to construct
    # a list of lists then concatenate them.
    if args.target:
        args.target = concat(args.target)

    args.target = args.target or [DEFAULT_TARGET]

    for i, target in enumerate(args.target):
        if target in TARGET_ALIASES:
            args.target[i] = TARGET_ALIASES[target]
        if "aarch64" in args.target[i] and not args.cross_compiler:
            args.cross_compiler = "aarch64-linux-gnu-g++"

    graph, lib, params, dumps = tvmc.compiler.compile_model(
        args.FILE,
        args.target,
        args.dump_codegen.split(","),
        "",
        args.language,
        args.sanitize_diagnostics,
        args.input_shape,
        args.tuner_file,
        args.alter_layout,
    )

    if dumps:
        tvmc.compiler.save_dumps(args.output, dumps)

    tvmc.compiler.save_module(args.output, graph, lib, params, args.cross_compiler)
    return 0


def drive_run(args):
    """ Invoke tvmc.runner module with command line arguments """

    inputs = {}
    if args.inputs:
        inputs = np.load(args.inputs)

    outputs, times = tvmc.runner.run_module(
        args.FILE,
        args.hostname,
        args.port,
        args.tracker_key,
        inputs=inputs,
        fill_mode=args.fill_mode,
        repeat=args.repeat,
        profile=args.profile,
    )
    if args.time:
        stat_table = tvmc.runner.format_times(times)
        print(stat_table)

    if args.print_top:
        top_results = tvmc.runner.get_top_results(outputs, args.print_top)
        print(top_results)

    if args.outputs:
        # Save the outputs
        np.savez(args.outputs, **outputs)

    return 0


def drive_tune(args):
    """ Invoke tvmc.autotuner module with command line arguments """

    # pylint: disable=C0415
    from tvm import autotvm

    if args.target in TARGET_ALIASES:
        args.target = TARGET_ALIASES[args.target]

    target_host = args.target

    mod, params = tvmc.frontends.load_model(args.FILE, args.language)

    tasks = tvmc.autotuner.get_tuning_tasks(
        mod,
        params,
        args.target,
        target_host=target_host,
        alter_layout=args.alter_layout,
    )

    if args.hostname:
        if not args.tracker_key:
            raise TVMCException(
                "need to provide tracker key (--tracker-key) for remote tuning"
            )

        runner = autotvm.RPCRunner(
            args.tracker_key,
            host=args.hostname,
            port=args.port,
            number=args.number,
            repeat=args.repeat,
            n_parallel=args.parallel,
            timeout=args.timeout,
            min_repeat_ms=args.min_repeat_ms,
        )
    else:
        runner = autotvm.LocalRunner(
            number=args.number,
            repeat=args.repeat,
            timeout=args.timeout,
            min_repeat_ms=args.min_repeat_ms,
        )

    tuning_option = {
        "tuner": args.tuner,
        "trials": args.trials,
        "early_stopping": args.early_stopping,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func="default"), runner=runner
        ),
    }

    tvmc.autotuner.tune_tasks(tasks, args.tuner_file, **tuning_option)
    return 0


def _main(argv):
    """ TVMC command line interface. """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="tvm compiler driver",
        epilog=__doc__,
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="increase verbosity"
    )
    parser.add_argument(
        "--version", action="store_true", help="print the version and exit"
    )
    subparsers = parser.add_subparsers(title="commands")

    add_compile_parser(subparsers)
    add_run_parser(subparsers)
    add_tune_parser(subparsers)

    args = parser.parse_args(argv)
    if args.verbose > 4:
        args.verbose = 4

    logging.getLogger().setLevel(40 - args.verbose * 10)

    if args.version:
        version = pkg_resources.get_distribution("tvmc").version
        sys.stdout.write("%s\n" % version)
        return 0

    if not hasattr(args, "func"):
        parser.error("missing subcommand")

    try:
        return args.func(args)
    except TVMCException as err:
        sys.stderr.write("error: %s\n" % err)
        return 4

def main():
    sys.exit(_main(sys.argv[1:]))

if __name__ == "__main__":
    main()
