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


"""Test compiling the on-device runtime."""
target = tvm.target.target.micro('stm32f746xx')

A = tvm.te.placeholder((2,), dtype='int8')
B = tvm.te.placeholder((1,), dtype='int8')
C = tvm.te.compute(A.shape, lambda i: A[i] + B[0], name='C')

s = tvm.te.create_schedule(C.op)

with tvm.transform.PassContext(opt_level=3, config={'tir.disable_vectorize': True}):
    mod = tvm.build(s, [A, B, C], target, target_host=target, name='add')

workspace = tvm.micro.Workspace(debug=True)

#rpc_session = tvm.rpc.connect('127.0.0.1', 9090)
project_dir = os.path.expanduser('~/git/utvm-zephyr-runtime')
compiler = zephyr.ZephyrCompiler(
    project_dir=project_dir,
    board='stm32f746g_disco',
    zephyr_toolchain_variant='gnuarmemb',
    env_vars={'GNUARMEMB_TOOLCHAIN_PATH': '/opt/gcc-arm-none-eabi-9-2019-q4-major'},
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

micro_binary = tvm.micro.build_static_runtime(workspace, compiler, mod, lib_opts, bin_opts)
lib_opts['cflags'].pop()

device_transport = device_util.DeviceTransportLauncher({
     'num_instances': 1,
     'use_tracker': False,
   })

with contextlib.ExitStack() as exit_stack:
    flasher_kw = {
      #'nrfjprog_snr': '960104913',
      #'debug': False,
      #'debug_remote_hostport': '{}:{:d}'.format(*device_transport.openocd_gdb_host_port_tuple(0)),
    }

flasher = compiler.Flasher(**flasher_kw, openocd_serial='066CFF313130525043233350')


micro_binary=tvm.micro.MicroBinary('/home/tgall/git/utvm-zephyr-runtime/__tvm_build', 'zephyr/zephyr.bin', debug_files=['zephyr/zephyr.elf'], labelled_files={'device_tree': ['zephyr/zephyr.dts'], 'cmake_cache': ['CMakeCache.txt']})

transport_context_manager=flasher.Transport(micro_binary)

sess=exit_stack.enter_context(tvm.micro.Session(micro_binary, flasher, transport_context_manager))

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

