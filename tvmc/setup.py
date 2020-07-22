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
from setuptools import setup


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="tvmc",
    description="TVM command line driver",
    long_description=readme(),
    long_description_content_type="text/markdown",
    packages=["tvmc"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.5",
    entry_points={"console_scripts": ["tvmc = tvmc.main:main"]},
    use_scm_version = {
        "root": "..",
        "relative_to": __file__,
        "local_scheme": "node-and-timestamp"
    },
    setup_requires=["flake8", "pytest-runner", "setuptools_scm<4"],
    install_requires=[
        "cython",
        "decorator",
        "flatbuffers",
        "grpcio<=1.27.2",
        "keras",
        "numpy>=1.14.5",
        "onnx",
        "pkgconfig",
        "scipy==1.4.1",
        "tensorflow==2.1.0",
        "tflite==2.1.0",
        "torch",
        "attrs",
        "xgboost",
        "tornado",
        "antlr4-python3-runtime",
    ],
    tests_require=["pytest"],
    test_suite="tests",
)
