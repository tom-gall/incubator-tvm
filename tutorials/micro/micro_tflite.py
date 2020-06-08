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
.. _tutorial-micro-tflite:

Micro TVM with TFLite Models
============================
**Author**: `Tom Gall <https://github.com/tom-gall>`_

This tutorial is an introduction to working with MicroTVM and TFLite models with Relay.
"""
######################################################################
# Setup
# -----
#
# To get started, TFLite package needs to be installed as prerequisite.
# 
# install tflite
# .. code-block:: bash
#
#   pip install tflite=2.1.0 --user
#
# or you could generate TFLite package yourself. The steps are the following:
#
#   Get the flatc compiler.
#   Please refer to https://github.com/google/flatbuffers for details
#   and make sure it is properly installed.
#
# .. code-block:: bash
#
#   flatc --version
#
# Get the TFLite schema.
#
# .. code-block:: bash
#
#   wget https://raw.githubusercontent.com/tensorflow/tensorflow/r1.13/tensorflow/lite/schema/schema.fbs
#
# Generate TFLite package.
#
# .. code-block:: bash
#
#   flatc --python schema.fbs
#
# Add current folder (which contains generated tflite module) to PYTHONPATH.
#
# .. code-block:: bash
#
#   export PYTHONPATH=${PYTHONPATH:+$PYTHONPATH:}$(pwd)
#
#
# To validate that the TFLite package was installed successfully, ``python -c "import tflite"``
#
# First we need to download a pretrained TFLite model. When working with microcontrollers
# you need to be mindful these are highly resource constrained devices as such standard 
# models like Mobilenet may not fit into their modest memory. 
#
# For this tutorial, we'll make use of one of the TF Micro example models.
# 
# If you wish to replicate the training steps see:
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples/hello_world/train
#
# .. code-block:: bash
#
# if you download the example pretrained model from
#   wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/micro/hello_world_2020_04_13.zip
#   unzip hello_world_2020_04_13.zip
#   this will fail due to an unimplemented opcode (114)
#   I've saved an older version of the pre-trailed model and made it available on linaro.org

######################################################################
# Python imports for tvm, numpy etc
# ----------------------------------------------
import os
import numpy as np
import tvm
import tvm.micro as micro
import dload

from tvm.contrib import graph_runtime, util
from tvm import relay


######################################################################
# Load the pretrained TFLite model from a file in your current 
# directory into a buffer
print (os.getcwd())
model_url = 'https://people.linaro.org/~tom.gall/sine_model.tflite'

model_file = dload.save(model_url)
print (model_file)
######################################################################
# Uncomment the following code to load the model from a local 
# directory
# Load the pretrained TFLite model from a file in your current 
# directory into a buffer
# model_dir ="./"
# tflite_model_file = os.path.join(model_dir, "sine_model.tflite")
tflite_model_buf = open(model_file, "rb").read()

######################################################################
# Using the buffer, transform into a tflite model python object
try:
    import tflite
    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

######################################################################
# Print out the version of the model
version = tflite_model.Version()
print ("Model Version: " + str(version))


######################################################################
# Setup the device config which is what will be used to communicate
# with the microcontroller (a STM32F746 Discovery board)
TARGET = 'c -device=micro_dev'
dev_config = micro.device.arm.stm32f746xx.generate_config("127.0.0.1", 6666)


######################################################################
# Parse the python model object to convert it into a relay module
# and weights
# It is important to note that the input tensor name must match what
# is contained in the model.
# If you are unsure what that might be, this can be discovered by using
# the visualize.py script within the Tensorflow project.
# See : How do I inspect a .tflite file? https://www.tensorflow.org/lite/guide/faq 
input_tensor = "dense_4_input"
input_shape = (1,)
input_dtype = "float32"

mod, params = relay.frontend.from_tflite(tflite_model,
                                         shape_dict={input_tensor: input_shape},
                                         dtype_dict={input_tensor: input_dtype})

######################################################################
# Next with the dev_config, we establish a micro session and create
# a context
with micro.Session(dev_config) as sess:
    ctx = tvm.micro_dev(0)

######################################################################
# Now we create a build config for relay. turning off two options
# and then calling relay.build which will result in a C source
    disable_vectorize = tvm.target.build_config(disable_vectorize=True)
    disable_fusion = relay.build_config(disabled_pass={'FuseOps'})
    with disable_vectorize, disable_fusion:
        graph, c_mod, params = relay.build(mod, target=TARGET, params=params)

######################################################################
# With the c_mod that is the handle to our c sourcecode, we create a
# micro module, followed by a compiled object which behind the scenes
# is linked to the microTVM runtime for running on the target board
    micro_mod = micro.create_micro_mod(c_mod, dev_config)
    mod = graph_runtime.create(graph, micro_mod, ctx)

######################################################################
# Pass the weights to get ready to do some inference
    mod.set_input(**params)

######################################################################
# The model consumes a single float32. Construct a tvm.nd.array object
# with a single contrived number as input. For this model values of 
# 0 to 2Pi are acceptible.
    mod.set_input(input_tensor, tvm.nd.array(np.array([0.5], dtype="float32")))


######################################################################
# Run the model ON DEVICE
# You'll need to uncomment this line for the example to work
#
# .. code-block:: python
#
#   mod.run()

######################################################################
# Get output from the run and print
# Uncomment the following two lines for the example to work,
#
# .. code-block:: python
#
#   tvm_output = mod.get_output(0).asnumpy()
#   print("result is: "+str(tvm_output))
