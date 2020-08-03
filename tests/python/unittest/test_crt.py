import contextlib
import copy
import glob
import os
import textwrap

import numpy

import tvm
import tvm.relay
import tvm.micro

from tvm.contrib import device_util
from tvm.contrib import util

DEBUG = False

def _setup(test_func):
  target = tvm.target.create('c -mcpu=x86-64')

  A = tvm.te.placeholder((2,), dtype='int8')
  B = tvm.te.placeholder((1,), dtype='int8')
  C = tvm.te.compute(A.shape, lambda i: A[i] + B[0], name='C')

  s = tvm.te.create_schedule(C.op)

  with tvm.transform.PassContext(opt_level=3, config={'tir.disable_vectorize': True}):
    mod = tvm.build(s, [A, B, C], target, target_host=target, name='add')

  workspace = tvm.micro.Workspace(debug=True)

  compiler = tvm.micro.DefaultCompiler(target=target)
  opts = tvm.micro.DefaultOptions()
  opts['include_dirs'].append(os.path.join(tvm.micro.TVM_ROOT_DIR, 'src', 'runtime', 'crt', 'host'))
  opts['include_dirs'].append(os.path.join(tvm.micro.TVM_ROOT_DIR, 'src', 'runtime', 'crt', 'include'))
  # lib_opts = copy.deepcopy(bin_opts)
  # lib_opts['profile']['common'].append('-Werror')
  # lib_opts['cflags'] = ['-Wno-error=incompatible-pointer-types']

  micro_binary = tvm.micro.build_static_runtime(workspace, compiler, mod, opts, opts)
#  device_transport = device_util.DeviceTransportLauncher({
#    'num_instances': 1,
#    'use_tracker': False,
#  })

  # with contextlib.ExitStack() as exit_stack:
  flasher_kw = {
    'debug': DEBUG,
  }

  flasher = compiler.Flasher(**flasher_kw)
  with tvm.micro.Session(binary=micro_binary, flasher=flasher) as sess:
    test_func(sess)


def test_compile_runtime(sess):
  """Test compiling the on-device runtime."""
  A_data = tvm.nd.array(numpy.array([2, 3], dtype='int8'), ctx=sess.context)
  B_data = tvm.nd.array(numpy.array([4], dtype='int8'), ctx=sess.context)
  C_data = tvm.nd.array(numpy.array([0, 0], dtype='int8'), ctx=sess.context)

  system_lib = sess.get_system_lib()
  print('got system lib', system_lib)
  system_lib.get_function('add')(A_data, B_data, C_data)
  system_lib.get_function('add')
  print('got data!', C_data.asnumpy())
  assert (C_data.asnumpy() == numpy.array([6, 7])).all()


def test_time_evaluator(sess):
  A_data = tvm.nd.array(numpy.array([2, 3], dtype='int8'), ctx=sess.context)
  B_data = tvm.nd.array(numpy.array([4], dtype='int8'), ctx=sess.context)
  C_data = tvm.nd.array(numpy.array([0, 0], dtype='int8'), ctx=sess.context)
  number = 10000
  repeat = 5
  min_repeat_ms = 100

  system_lib = sess.get_system_lib()
  timer_func = system_lib.time_evaluator(
    'add', sess.context,
    number=number, repeat=repeat, min_repeat_ms=min_repeat_ms)
  time_res = timer_func(A_data, B_data, C_data)
  assert len(time_res.results) == repeat
  assert time_res.mean > 0.0
  assert (C_data.asnumpy() == numpy.array([6, 7])).all()

  # run the time evaluator many times to ensure we don't leak memory (we
  # allocate a byte array each time to store the doubles)
  timer_func = system_lib.time_evaluator(
    'add', sess.context,
    number=1, repeat=5)
  for i in range(50):
    print(timer_func(A_data, B_data, C_data))


def test_autotvm():
  target = tvm.target.create('c -mcpu=x86-64')
  model = tvm.relay.fromtext("""\
      v0.0.4
      def @main(%data: Tensor[(1, 3, 32, 32), int8], %weight: Tensor[(3, 3, 5, 5), int8]) {
          nn.conv2d(
               %data,
               %weight,
               padding=[2, 2],
               channels=3,
               kernel_size=[5, 5],
               data_layout="NCHW",
               kernel_layout="OIHW",
               out_dtype="int32")

      }
      """)
  with tvm.transform.PassContext(opt_level=3, config={'tir.disable_vectorize': True}):
    tasks = tvm.autotvm.task.extract_from_program(model, {}, target)

  assert len(tasks) == 1
  tuner = tvm.autotvm.tuner.GATuner(tasks[0])

  workspace = tvm.micro.Workspace(debug=True)
  compiler = tvm.micro.DefaultCompiler(target=target)
  lib_opts = tvm.micro.DefaultOptions()
  lib_opts['include_dirs'].append(os.path.join(tvm.micro.TVM_ROOT_DIR, 'src', 'runtime', 'crt', 'host'))
  lib_opts['include_dirs'].append(os.path.join(tvm.micro.TVM_ROOT_DIR, 'src', 'runtime', 'crt', 'include'))

  adapter = tvm.micro.AutoTvmAdapter(workspace, compiler, compiler.flasher_factory,
                                     lib_opts=lib_opts)

  builder = tvm.autotvm.LocalBuilder(
      build_func=adapter.StaticRuntime,
      n_parallel=1,
      build_kwargs={'build_option': {'tir.disable_vectorize': True}},
      do_fork=False)
  runner = tvm.autotvm.LocalRunner(
      number=1, repeat=1, timeout=0, code_loader=adapter.CodeLoader)

  measure_option = tvm.autotvm.measure_option(builder=builder, runner=runner)
  tuner.tune(n_trial=5,
             measure_option=measure_option,
             callbacks=[],
             si_prefix='k')




if __name__ == '__main__':
  _setup(test_compile_runtime)
  _setup(test_time_evaluator)
  import logging
  logging.basicConfig(level=logging.DEBUG)
  test_autotvm()
