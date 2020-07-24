# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#_log2
#   http://0.apache.0 << org_log2/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
.. _tutorial-micro-keras:

Running a Keras model on microTVM
=================================
**Author**: `Andrew Reusch <https://github.com/areusch>`_

This tutorial explains how to import a Keras model into TVM, compile it, and execute it on any
microTVM-compatible device.
"""

####################
# Defining the model
####################
#
# To begin with, define a model in Keras to be executed on-device. This shouldn't look any different
# from a usual Keras model definition. Let's define a relatively small model here for efficiency's
# sake.


import tensorflow as tf
import keras

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(4,  3))
model.build(input_shape=(None, 32, 32, 3))
model.summary()


####################
# Importing into TVM
####################
# Now, use `from_keras <https://tvm.apache.org/docs/api/python/relay/frontend.html#tvm.relay.frontend.from_keras>`_ to import the Keras model into TVM.


import tvm
from tvm import relay
inputs = {i.name.split(':', 2)[0]: [x if x is not None else 1 for x in i.shape.as_list()]
          for i in model.inputs}
tvm_model, params = relay.frontend.from_keras(model, inputs, layout='NHWC')


# %%
# ``from_keras`` produces two artifacts: an `IRModule <https://tvm.apache.org/docs/api/python/ir.html?highlight=irmodule#tvm.ir.IRModule>`_  and a dictionary of extracted parameters.


import pprint
pprint.pprint(params)


# %%
# We can now inspect the generated Relay code. An ``IRModule`` is a collection of related functions
# so first let's take a look at the names of the generated functions.


print([f.name_hint for f, _ in tvm_model.functions.items()])


# %%
# We can also look at the Relay source for a particular function. To do so, we simply lookup the
# Function object by name and convert it to `str`. By TVM convention, the top-level function
# (and in this case, the only function) is ``main``.

print(str(tvm_model['main']))


##########################
# Running the TVM Compiler
##########################
# Now that the model has been imported into Relay, let's run the TVM compiler to produce optimized C
# source code. Then we'll compile that source code for our microTVM target. First, we'll define our
# TVM target, which is the C backend targeting x86-64 in this case.


target = tvm.target.create('c -mcpu=x86-64')


# %%
# Next we'll set compiler options and run the compiler.
#
# Note that due to a limitation of the ``c`` backend, vectorization needs to be disabled for
# microTVM.


with tvm.transform.PassContext(opt_level=3, config={'tir.disable_vectorize': True}):
    graph, lowered_module, lowered_params = relay.build(tvm_model, target=target, params=params)


# %%
# Let's take a look at what we have now. First, TVM has produced an execution graph defining the
# low-level C functions to invoke, any tensors used by those functions, and the order in which to
# invoke them:


print(graph)


# %%
# Next, the ``lowered_module`` contains C source code to be compiled by the cross-compiler. In this
# case, since we're executing on x86, this is just plain old `gcc`.


print(lowered_module.get_source())


# %%
# Finally, the parameters have been renamed as part of the lowering process. As optimizations such
# as constant folding are introduced, some parameters may change or become unnecessary. Be sure to
# use these ``lowered_params`` with your code.


pprint.pprint(lowered_params)


############################
# Building a microTVM binary
############################
# For microTVM, we need to compile and link 3 pieces together:
#
# #. The compiled model (which we have as ``lowered_module``).
# #. The CRT runtime and RPC server (from ``src/runtime/crt``)
# #. The main entry point and top-level configuration (from ``src/runtime/crt/host``).
#
# You can use your own build process for this (see ``src/runtime/crt/Makefile`` for an example) by
# exporting ``lowered_module`` as C::
#     lowered_module.save('path/to/module.c', 'cc')
#
# However, for the purposes of this guide, we'll just re-use the AutoTVM compilation infrastructure:

import os
import tvm.micro

workspace = tvm.micro.Workspace(debug=True)
compiler = tvm.micro.DefaultCompiler(target)

repo_root = os.path.abspath(os.path.join('..', '..'))
lib_opts = {
  'include_dirs': [
    f'{repo_root}/src/runtime/crt/host',
    f'{repo_root}/src/runtime/crt/include',
    f'{repo_root}/include',
    f'{repo_root}/3rdparty/dlpack/include',
    f'{repo_root}/3rdparty/dmlc-core/include',
    f'{repo_root}/3rdparty/mbed-os/targets/TARGET_NORDIC/TARGET_NRF5x/TARGET_SDK_11/libraries/crc16/',
  ],
  'ccflags': ['-std=c++11'],
}
micro_binary = tvm.micro.build_static_runtime(workspace, compiler, lowered_module, lib_opts, lib_opts)


##################################
# Running the model under microTVM
##################################
# Now we should have built an ELF executable. For the purposes of this guide, we'll execute the compiled
# binary and send RPC traffic through `stdin` and `stdout`. Should you target a microcontroller, you'd
# instead *flash* the compiled binary onto the device and communicate using e.g. a UART. The AutoTVM API
# reflects this expected use case.
#
# Now, let's build some random inputs and run inference.


from tvm.contrib import graph_runtime
import numpy as np
input_data = {k: (np.random.rand(*shape) / 1e3).astype(np.float32) for k, shape in inputs.items()}


flasher = compiler.Flasher()
with tvm.micro.Session(binary=micro_binary, flasher=flasher) as sess:
  graph_rt = graph_runtime.create(graph, sess.get_system_lib(), sess.context)
  nd_arrs = dict(lowered_params)
  for key, np_arr in input_data.items():
    nd_arrs[key] = tvm.runtime.ndarray.array(np_arr, ctx=sess.context)
  graph_rt.run(**nd_arrs)

  output = graph_rt.get_output(0).asnumpy()

print(output)


# %%
# Great! We've now compiled and executed our model on x86 using the microTVM runtime.
#


##########################
# Verifying model accuracy
##########################
# Finally, it's important to make sure that the model we've executed is producing accurate results.
# There are many such strategies for verification. Here, we'll demonstrate how to randomly sample
# input tensors and compare the output tensor within a tolerance to a reference implementation.
#
# For this tutorial, we'll use Keras as our reference implementation.
for i in range(10):
  with tvm.micro.Session(binary=micro_binary, flasher=flasher) as sess:
    graph_rt = graph_runtime.create(graph, sess.get_system_lib(), sess.context)
    input_data = {k: (np.random.rand(*shape) / 1e3).astype(np.float32) for k, shape in inputs.items()}
    nd_arrs = dict(lowered_params)
    for key, np_arr in input_data.items():
      nd_arrs[key] = tvm.runtime.ndarray.array(np_arr, ctx=sess.context)
    graph_rt.run(**nd_arrs)

    utvm_output = graph_rt.get_output(0).asnumpy()
    keras_output = model.predict(input_data)
    np.testing.assert_allclose(utvm_output, keras_output, atol=1e-5)
    print(f'iteration {i}: all matched')


# %%
##################
# Closing thoughts
##################
# This tutorial shows how TVM can be easily leveraged to evaluate Keras models under a minimal runtime.
# Here are some other considerations worth thinking through when building models for microTVM targets.
#
#
# #. Device resources are often limited. Ensure input, output, and intermediate tensors will fit in device
#    RAM.
# #. Debugging can be complex. microTVM defines the ``TVMLogf`` function for printf-style debugging over
#    the RPC socket. You can use this when implementing your own functions or adapting the microTVM
#    runtime.
# #. Quanitzation can produce good performance improvements, but must be tailored to the model in use.
