import contextlib
import copy
import glob
import os
import shutil

import numpy

import tvm
import tvm.micro
import tvm.relay

from tvm.micro.contrib import zephyr
from tvm.contrib import device_util
from tvm.contrib import util

BUILD = True
DEBUG = False

def test_compile_runtime():
  """Test compiling the on-device runtime."""
#  target = tvm.target.target.micro('stm32f746xx')
  target = tvm.target.target.micro('nrf5340')

  A = tvm.te.placeholder((2,), dtype='int8')
  B = tvm.te.placeholder((1,), dtype='int8')
  C = tvm.te.compute(A.shape, lambda i: A[i] + B[0], name='C')

  s = tvm.te.create_schedule(C.op)

  with tvm.transform.PassContext(opt_level=3, config={'tir.disable_vectorize': True}):
    mod = tvm.build(s, [A, B, C], target, target_host=target, name='add')

  workspace = tvm.micro.Workspace(debug=True)

#  rpc_session = tvm.rpc.connect('127.0.0.1', 9090)
  project_dir = os.path.expanduser('~/ws/utvm-zephyr-runtime')
  compiler = zephyr.ZephyrCompiler(
    project_dir=project_dir,
#    board='nucleo_f746zg',
    board='nrf5340pdk_nrf5340_cpuapp',
    zephyr_toolchain_variant='gnuarmemb',
    env_vars={'GNUARMEMB_TOOLCHAIN_PATH': '~/ws/gcc-arm-none-eabi-9-2020-q2-update'},
  )

  root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
  bin_opts = tvm.micro.DefaultOptions()
  bin_opts.setdefault('profile', {})['common'] = ['-Wno-unused-variable']
#  bin_opts.setdefault('ccflags', []).append('-std=gnu++14')
  bin_opts.setdefault('ldflags', []).append('-std=gnu++14')
  bin_opts.setdefault('include_dirs', []).append(f'{project_dir}/crt')
  bin_opts.setdefault('include_dirs', []).append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'runtime', 'crt', 'include')))

  lib_opts = copy.deepcopy(bin_opts)
  lib_opts['profile']['common'].append('-Werror')
  lib_opts['cflags'] = ['-Wno-error=incompatible-pointer-types']

  if BUILD:
    micro_binary = tvm.micro.build_static_runtime(workspace, compiler, mod, lib_opts, bin_opts)
  lib_opts['cflags'].pop()

#   device_transport = device_util.DeviceTransportLauncher({
#     'num_instances': 1,
#     'use_tracker': False,
#   })

  with contextlib.ExitStack() as exit_stack:
    flasher_kw = {
       'nrfjprog_snr': '960104913',
#      'debug': DEBUG,
#      'debug_remote_hostport': '{}:{:d}'.format(*device_transport.openocd_gdb_host_port_tuple(0)),
    }

#     if DEBUG:
#       flasher_kw['debug_wrapping_context_manager'] = device_transport.launch(None)

    flasher = compiler.Flasher(**flasher_kw, openocd_serial='066DFF343039564157214347')
    if BUILD:
      sess = exit_stack.enter_context(tvm.micro.Session(binary=micro_binary, flasher=flasher))
    else:
      sess = exit_stack.enter_context(tvm.micro.Session(transport_context_manager=flasher.Transport(
        micro_binary=tvm.micro.MicroBinary('/private/var/folders/9y/3j808g591ln3kys4qpyl3qmc0000gn/T/tvm-debug-mode-tempdirs/2020-06-30T09-34-07___s8a9ul7f/00000/build/runtime', 'zephyr/zephyr.bin', debug_files=['zephyr/zephyr.elf'], labelled_files={'device_tree': ['zephyr/zephyr.dts'], 'cmake_cache': ['CMakeCache.txt']}))))
    A_data = tvm.nd.array(numpy.array([2, 3], dtype='int8'), ctx=sess.context)
    assert (A_data.asnumpy() == numpy.array([2, 3])).all()

    B_data = tvm.nd.array(numpy.array([4], dtype='int8'), ctx=sess.context)
    assert (B_data.asnumpy() == numpy.array([4])).all()

    C_data = tvm.nd.array(numpy.array([0, 0], dtype='int8'), ctx=sess.context)
    assert (C_data.asnumpy() == numpy.array([0, 0])).all()

    print('get system lib')
    system_lib = sess._rpc.system_lib()
#    system_lib_func = sess._rpc.get_function('get_system_lib')
#    system_lib = system_lib_func()
    print('got system lib', system_lib)
    system_lib.get_function('add')(A_data, B_data, C_data)
    print('got data!', C_data.asnumpy())
    assert (C_data.asnumpy() == numpy.array([6, 7])).all()

def test_autotvm():
  target = tvm.target.target.micro('nrf5340')
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

  project_dir = os.path.expanduser('~/ws/nrf5340/autotvm')
  compiler = zephyr.ZephyrCompiler(
    project_dir=project_dir,
#    board='nucleo_f746zg',
    board='nrf5340pdk_nrf5340_cpuapp',
    zephyr_toolchain_variant='gnuarmemb',
    env_vars={'GNUARMEMB_TOOLCHAIN_PATH': '/usr/local'},
    zephyr_base=os.path.expanduser('~/ws/nrf5340/zephyr/zephyr'),
  )

  root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
  bin_opts = tvm.micro.DefaultOptions()
  bin_opts.setdefault('profile', {})['common'] = ['-Wno-unused-variable']
#  bin_opts.setdefault('ccflags', []).append('-std=gnu++14')
  bin_opts.setdefault('ldflags', []).append('-std=gnu++14')
  bin_opts.setdefault('include_dirs', []).append(f'{project_dir}/crt')
  bin_opts.setdefault('include_dirs', []).append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'runtime', 'crt', 'include')))

  lib_opts = copy.deepcopy(bin_opts)
  lib_opts['profile']['common'].append('-Werror')
  lib_opts['cflags'] = ['-Wno-error=incompatible-pointer-types']

  adapter = tvm.micro.AutoTvmAdapter(workspace, compiler, compiler.flasher_factory.override_kw(nrfjprog_snr='960100695'),
                                     lib_opts=lib_opts, bin_opts=bin_opts)

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
#  test_compile_runtime()
  import logging
  logging.basicConfig(level=logging.DEBUG)
  test_autotvm()
