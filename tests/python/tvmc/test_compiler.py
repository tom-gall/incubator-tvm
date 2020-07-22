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
import os
from os import path

import tvmc

print(tvmc.__file__)
print(dir(tvmc))

#from .tvmc_fixtures import keras_resnet50
#from .tvmc_fixtures import onnx_resnet50
#from .tvmc_fixtures import tflite_mobilenet_v1_1_quant

def test_save_dumps(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("data")
    dump_formats = {"relay": "fake relay", "ll": "fake llvm", "asm": "fake asm"}
    tvmc.compiler.save_dumps("fake_module", dump_formats, dump_root=tmpdir)

    assert path.exists("{}/{}".format(tmpdir, "fake_module.ll"))
    assert path.exists("{}/{}".format(tmpdir, "fake_module.asm"))
    assert path.exists("{}/{}".format(tmpdir, "fake_module.relay"))


# End to end tests for compilation


def test_compile_tflite_module(tflite_mobilenet_v1_1_quant):
    graph, lib, params, dumps = tvmc.compiler.compile_model(
        tflite_mobilenet_v1_1_quant,
        targets=["llvm"],
        dump_sources=["ll"],
        alter_layout="NCHW",
    )


def test_cross_compile_aarch64_tflite_module(tflite_mobilenet_v1_1_quant):
    graph, lib, params, dumps = tvmc.compiler.compile_model(
        tflite_mobilenet_v1_1_quant,
        targets=["llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon"],
        dump_sources=["asm"],
    )


def test_compile_keras__save_module(keras_resnet50, tmpdir_factory):
    graph, lib, params, dumps = tvmc.compiler.compile_model(
        keras_resnet50, targets=["llvm"], dump_sources=["ll"]
    )

    module_file = os.path.join(tmpdir_factory.mktemp("saved_output"), "saved.tar")
    tvmc.compiler.save_module(module_file, graph, lib, params)


def test_cross_compile_aarch64_keras_module(keras_resnet50):
    graph, lib, params, dumps = tvmc.compiler.compile_model(
        keras_resnet50,
        targets=["llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon"],
        dump_sources=["asm"],
    )


def test_compile_onnx_module(onnx_resnet50):
    graph, lib, params, dumps = tvmc.compiler.compile_model(
        onnx_resnet50, targets=["llvm"], dump_sources=["ll"]
    )


def test_cross_compile_aarch64_onnx_module(onnx_resnet50):
    graph, lib, params, dumps = tvmc.compiler.compile_model(
        onnx_resnet50,
        targets=["llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon"],
        dump_sources=["asm"],
    )
