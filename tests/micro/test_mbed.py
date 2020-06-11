import contextlib
import copy
import glob
import os

import numpy

import tvm
import tvm.micro

from tvm.micro.contrib import mbed
from tvm.contrib import device_util
from tvm.contrib import util

DEBUG = False

def test_compile_runtime():
  """Test compiling the on-device runtime."""
  target = tvm.target.target.micro('stm32f746xx')
#  target = tvm.target.create('c -mcpu=x86-64')

  A = tvm.te.placeholder((2,), dtype='int8')
  B = tvm.te.placeholder((1,), dtype='int8')
  C = tvm.te.compute(A.shape, lambda i: A[i] + B[0], name='C')

  s = tvm.te.create_schedule(C.op)

  with tvm.transform.PassContext(opt_level=3, config={'tir.disable_vectorize': True}):
    mod = tvm.build(s, [A, B, C], target, target_host=target, name='add')

  workspace = tvm.micro.Workspace(debug=True)

  project_dir = '/Users/andrew/ws/stm-nucleo/test'
#  compiler = tvm.micro.DefaultCompiler(target=target)
  compiler = mbed.MbedCompiler(
    project_dir=project_dir,
    mbed_target='NUCLEO_F746ZG',
    mbed_toolchain='GCC_ARM')

  root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
  bin_opts = {
    'profile': {'common': ['-Wno-unused-variable']},
    'ccflags': ['-std=gnu++14'],  # Need gnu++14 for https://github.com/ARMmbed/mbed-os/issues/10548
    'ldflags': ['-std=gnu++14'],
    'include_dirs': [f'{root_dir}/include',
                     f'{root_dir}/3rdparty/dlpack/include',
                     f'{root_dir}/3rdparty/mbed-os/targets/TARGET_NORDIC/TARGET_NRF5x/TARGET_SDK_11/libraries/crc16/',
                     f'{project_dir}/crt',
                     f'{root_dir}/3rdparty/dmlc-core/include'],
  }

  lib_opts = copy.deepcopy(bin_opts)
  lib_opts['profile']['common'].append('-Werror')
  lib_opts['cflags'] = ['-Wno-error=incompatible-pointer-types']

  micro_binary = tvm.micro.build_static_runtime(workspace, compiler, mod, lib_opts, bin_opts)
  lib_opts['cflags'].pop()

  device_transport = device_util.DeviceTransportLauncher({
    'num_instances': 1,
    'use_tracker': False,
  })

  with contextlib.ExitStack() as exit_stack:
    flasher_kw = {
      'debug': DEBUG,
      'debug_remote_hostport': '{}:{:d}'.format(*device_transport.openocd_gdb_host_port_tuple(0)),
    }

    if DEBUG:
      flasher_kw['debug_wrapping_context_manager'] = device_transport.launch(None)

    flasher = compiler.Flasher(**flasher_kw)
    with tvm.micro.Session(binary=micro_binary, flasher=flasher) as sess:
      A_data = tvm.nd.array(numpy.array([2, 3], dtype='int8'), ctx=sess.context)
      B_data = tvm.nd.array(numpy.array([4], dtype='int8'), ctx=sess.context)
      C_data = tvm.nd.array(numpy.array([0, 0], dtype='int8'), ctx=sess.context)

      system_lib = sess._rpc.system_lib()
#    system_lib_func = sess._rpc.get_function('get_system_lib')
#    system_lib = system_lib_func()
      print('got system lib', system_lib)
      system_lib.get_function('add')(A_data, B_data, C_data)
      print('got data!', C_data.asnumpy())
      assert C_data.asnumpy() == numpy.array([6, 7])


if __name__ == '__main__':
  test_compile_runtime()
