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
import pytest
import os
import tarfile

from tvm import autotvm

import tvmc.autotuner

from os import path

#from .tvmc_fixtures import onnx_resnet50


def _get_tasks(model):
    mod, params = tvmc.frontends.load_model(model)
    return tvmc.autotuner.get_tuning_tasks(mod, params, "llvm")


def _get_measure_options():
    return autotvm.measure_option(builder=autotvm.LocalBuilder(build_func="default"), runner='local')


def _tuner_test_helper(model, tuner_name, tmpdir_name, tasks=None, early_stopping=1, transfer_learning=True):
    tasks = tasks if tasks else _get_tasks(model)
    log_file = os.path.join(tmpdir_name, 'log_{}.txt'.format(tuner_name))

    tvmc.autotuner.tune_tasks(tasks=[tasks[0]],
                              log_file=log_file,
                              measure_option=_get_measure_options(),
                              tuner=tuner_name,
                              trials=1,
                              early_stopping=early_stopping,
                              use_transfer_learning=transfer_learning)

    # testing whether the log file was produced
    assert path.exists(log_file)


def test_get_tuning_tasks(onnx_resnet50):
    sut = _get_tasks(onnx_resnet50)
    expected_task_type = autotvm.task.Task

    assert type(sut) is list
    assert len(sut) > 0
    assert all([type(x) is expected_task_type for x in sut]) is True


def test_tune_tasks__tuner__xgb(onnx_resnet50, tmpdir_factory):
    tmpdir_name = tmpdir_factory.mktemp("data")
    _tuner_test_helper(onnx_resnet50, "xgb", tmpdir_name)


def test_tune_tasks__tuner__xgb_knob(onnx_resnet50, tmpdir_factory):
    tmpdir_name = tmpdir_factory.mktemp("data")
    _tuner_test_helper(onnx_resnet50, "xgb_knob", tmpdir_name)


def test_tune_tasks__tuner__ga(onnx_resnet50, tmpdir_factory):
    tmpdir_name = tmpdir_factory.mktemp("data")
    _tuner_test_helper(onnx_resnet50, "ga", tmpdir_name)


def test_tune_tasks__tuner__random(onnx_resnet50, tmpdir_factory):
    tmpdir_name = tmpdir_factory.mktemp("data")
    _tuner_test_helper(onnx_resnet50, "random", tmpdir_name)


def test_tune_tasks__tuner__gridsearch(onnx_resnet50, tmpdir_factory):
    tmpdir_name = tmpdir_factory.mktemp("data")
    _tuner_test_helper(onnx_resnet50, "gridsearch", tmpdir_name)


def test_tune_tasks__tuner__gridsearch__transfer_learning(onnx_resnet50, tmpdir_factory):
    tmpdir_name = tmpdir_factory.mktemp("data")
    _tuner_test_helper(onnx_resnet50, "gridsearch", tmpdir_name)

    # Exercises transfer learning by making sure a previous log exists,
    # it should be automatically picked up by the tuner.
    _tuner_test_helper(onnx_resnet50, "gridsearch", tmpdir_name)


def test_tune_tasks__tuner__ga__empty_tasks(onnx_resnet50, tmpdir_factory):
    tmpdir_name = tmpdir_factory.mktemp("data")
    _tuner_test_helper(onnx_resnet50, "ga", tmpdir_name, tasks=[])


def test_tune_tasks__tuner__xgb__no_early_stopping(onnx_resnet50, tmpdir_factory):
    tmpdir_name = tmpdir_factory.mktemp("data")
    _tuner_test_helper(onnx_resnet50, "xgb", tmpdir_name, tasks=None, early_stopping=None)


def test_tune_tasks__tuner__xgb__no_transfer_learning(onnx_resnet50, tmpdir_factory):
    tmpdir_name = tmpdir_factory.mktemp("data")
    _tuner_test_helper(onnx_resnet50, "xgb", tmpdir_name, tasks=None, transfer_learning=False)


def test_tune_tasks__invalid_tuner(onnx_resnet50, tmpdir_factory):
    tasks = _get_tasks(onnx_resnet50)
    log_file = os.path.join(tmpdir_factory.mktemp("data"), 'log2.txt')

    with pytest.raises(ValueError) as e:
        def f():
            tvmc.autotuner.tune_tasks(tasks, log_file, _get_measure_options(), "invalid_tuner", 1, 1)
        f()

    assert 'Invalid tuner:' in str(e.value)
