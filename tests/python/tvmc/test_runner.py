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
import numpy as np

import tvmc.compiler
import tvmc.runner

from tvmc.common import TVMCException

# from .tvmc_fixtures import tflite_compiled_module_as_tarfile

def test_generate_tensor_data_zeros():
    expected_shape = (2, 3)
    expected_dtype = "uint8"
    sut = tvmc.runner.generate_tensor_data(expected_shape, expected_dtype, "zeros")

    assert sut.shape == (2, 3)


def test_generate_tensor_data_ones():
    expected_shape = (224, 224)
    expected_dtype = "uint8"
    sut = tvmc.runner.generate_tensor_data(expected_shape, expected_dtype, "ones")

    assert sut.shape == (224, 224)


def test_generate_tensor_data_random():
    expected_shape = (2, 3)
    expected_dtype = "uint8"
    sut = tvmc.runner.generate_tensor_data(expected_shape, expected_dtype, "random")

    assert sut.shape == (2, 3)


def test_generate_tensor_data__type_float():
    expected_shape = (2, 3)
    expected_dtype = "float32"
    sut = tvmc.runner.generate_tensor_data(expected_shape, expected_dtype, "random")

    assert sut.shape == (2, 3)


def test_generate_tensor_data__type_unknown():
    with pytest.raises(TVMCException) as e:
        def f():
            tvmc.runner.generate_tensor_data((2, 3), "float32", "not_quite_random")
        f()

    assert 'unknown fill-mode: not_quite_random' in str(e.value)


def test_format_times_outputs_string():
    sut = tvmc.runner.format_times([60, 120, 12, 42])
    assert type(sut) is str


def test_format_times__contains_header():
    sut = tvmc.runner.format_times([60, 120, 12, 42])
    assert 'std (s)' in sut


def test_get_top_results_keep_results():
    fake_outputs = { 'output_0': np.array([[1, 2, 3, 4], [5, 6, 7, 8]]) }
    number_of_results_wanted = 3
    sut = tvmc.runner.get_top_results(fake_outputs, number_of_results_wanted)

    expected_number_of_lines = 2
    assert len(sut) == expected_number_of_lines

    expected_number_of_results_per_line = 3
    assert len(sut[0]) == expected_number_of_results_per_line
    assert len(sut[1]) == expected_number_of_results_per_line


def test_get_top_results_keep_results__limit_bigger_than_returned():
    fake_outputs = { 'output_0': np.array([[1, 2, 3, 4], [5, 6, 7, 8]]) }
    number_of_results_wanted = 6
    sut = tvmc.runner.get_top_results(fake_outputs, number_of_results_wanted)

    expected_number_of_lines = 2
    assert len(sut) == expected_number_of_lines

    # despite 'number_of_results_wanted' being 6,
    # we limit to the maximum available, in this
    # case it is 4.
    expected_number_of_results_per_line = 4
    assert len(sut[0]) == expected_number_of_results_per_line
    assert len(sut[1]) == expected_number_of_results_per_line

def test_run_tflite_module(tflite_compiled_module_as_tarfile):
    outputs, times = tvmc.runner.run_module(tflite_compiled_module_as_tarfile,
                                            hostname=None,
                                            fill_mode="zeros")

def test_run_tflite_module__with_profile(tflite_compiled_module_as_tarfile):
    outputs, times = tvmc.runner.run_module(tflite_compiled_module_as_tarfile,
                                            hostname=None,
                                            fill_mode="ones",
                                            profile=True)
