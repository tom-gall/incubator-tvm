<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# TVMC

```tvmc``` is a tool that provides useful command line invocations to compile,
run and tune models using TVM graph runtime.

In order to compile and tune, ```tvmc``` takes a model file and parameters as inputs,
and outputs a TAR file that contains the TVM modules that represent the
input model, graph and weights, for the required target. Target can be native or
cross-compiled.

When running a given model, ```tvmc``` expects a compiled model and input tensor values, so
that it can produce the outputs, when running on the required target, local or remote.

This document presents an overview and a short tutorial about ```tvmc```.

## Installation

```tvmc``` is a Python tool and - provided TVM and dependencies are available - it can be
installed in various ways.

The recommended way to install ```tvmc``` is via it's ```setuptools``` configuration file,
located at ```tvm/tvmc/setup.py```. To do that, go to the the TVM directory and run the
installation command, as described below:

    cd tvm/tvmc
    python setup.py install

The command above should install everything needed to get started with ```tvmc```, including
all the the supported frontends.

Once ```tvmc``` is installed, the main entry-point is the ```tvmc``` command line. A set of
sub-commands are available, to run the specific tasks offered by ```tvmc```: ```tune```,
```compile``` and ```run```.

The simplest way to get more information about a specific sub-command is ```tvmc <subcommand>
-- help```.

    tvmc compile --help

##  Usage

Now, let's compile a network and generate a few predictions using ```tvmc```.

As described above, in order to compile a model using ```tvmc```, the first thing we need is
a model file. For the sake of this example, let's use a MobileNet V1 model, in TFLite format.
More information about the model is available on
[this page](https://www.tensorflow.org/lite/guide/hosted_models).

To download and un-compress the ```.tgz``` file (34Mb), so that we can access the TFLite model,
run the command lines below:

    wget https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz
    tar xvzf mobilenet_v1_1.0_224_quant.tgz

With these commands, we should be able to provide the MobileNet V1 file (```mobilenet_v1_1.0_224_quant.tflite```)
to ```tvmc```, and obtain our TVM compiled model as an output. To do that, run the
following command line:

    tvmc compile -v mobilenet_v1_1.0_224_quant.tflite -o compiled_model.tar

As an output, you will notice a ```compiled_model.tar```, in the same directory.

Now it is time to feed the model with some input, that will generate a prediction using TVM.
As models are very diverse in terms of input formats and the source of those inputs (images, streams,
sensors, sound, to name a few), ```tvmc``` supports ```.npz``` (serialized NumPy arrays) as the
main format for ```tvmc run```. To learn more about the ```.npz``` format, please read the
[documentation](https://numpy.org/doc/stable/reference/generated/numpy.savez.html) on NumPy website.

MobileNet V1 expects a ```(224, 224, 3)``` input tensor. The Python code snippet below, can be used
as an example on how to convert a PNG file into a ```.npz``` file in the expected shape.
The example below uses [PIL](https://pillow.readthedocs.io/en/stable/) and
[NumPy](https://numpy.org) functions to import the image and generate the expected file.

    from tvm.contrib.download import download_testdata
    from PIL import Image
    import numpy as np

    cat_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
    image_path = download_testdata(cat_url, 'imagenet_cat.png', module='data')
    resized_image = Image.open(image_path).resize((224, 224))
    image_data = np.asarray(resized_image).astype("float32")
    image_data = np.expand_dims(image_data, axis=0)

    np.savez("imagenet_cat", input=image_data)

The snippet above will created a file named ```imagenet_cat.npz```, that we can use as an input for
```tvmc run```. The command is shown below:

    tvmc run --inputs cat.npz --output output_tensor.npz compiled_model.tar

The output for ```tvmc run``` is the exact output tensors for the compiled model. The output from our
example looks like the following string:

    [[283 282 286 464 264]
     [135  50  36   8   5]]

The numbers on the top line represent MobileNet classes, that can be checked in
[this file](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/demo/app/src/main/assets/labels_mobilenet_quant_v1_224.txt),
where each line represents a MobileNet class. The numbers on second line, are proportional
to the probability that the image is of the corresponding class.

## Tests

The set of tests that validate ```tvmc``` are located at ```tvm/tests/python/tvmc```. The helper
script ```tests/scripts/task_python_tvmc.sh``` can be used to trigger the all tests.
