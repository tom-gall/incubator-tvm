import abc
import atexit
import contextlib
import logging
import os
import signal
import string
import subprocess
import time
import typing

import serial
import serial.tools.list_ports


_LOG = logging.getLogger(__name__)


class SerialPortNotFoundError(Exception):
  """Raised when SerialSession cannot find the serial port specified."""


class Transport(metaclass=abc.ABCMeta):

    def __enter__(self):
      self.open()
      return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
      self.close()

    @abc.abstractmethod
    def open(self):
      raise NotImplementedError()

    @abc.abstractmethod
    def close(self):
      raise NotImplementedError()

    @abc.abstractmethod
    def read(self, n):
      raise NotImplementedError()

    @abc.abstractmethod
    def write(self, data):
      raise NotImplementedError()


class TransportLogger(Transport):

  def __init__(self, name, child, logger=None, level=logging.INFO):
    self.name = name
    self.child = child
    self.logger = logger or _LOG
    self.level = level

  # Construct PRINTABLE to exclude whitespace from string.printable.
  PRINTABLE = (string.digits + string.ascii_letters + string.punctuation)

  @classmethod
  def _to_hex(cls, data):
    lines = []
    if not data:
      lines.append('')
      return lines

    for i in range(0, (len(data) + 15) // 16):
      chunk = data[i * 16:(i + 1) * 16]
      hex_chunk = ' '.join(f'{c:02x}' for c in chunk)
      ascii_chunk = ''.join((chr(c) if chr(c) in cls.PRINTABLE else '.') for c in chunk)
      lines.append(f'{i * 16:04x}  {hex_chunk:47}  {ascii_chunk}')

    if len(lines) == 1:
      lines[0] = lines[0][6:]

    return lines

  def open(self):
    self.logger.log(self.level, 'opening transport')
    self.child.open()

  def close(self):
    self.logger.log(self.level, 'closing transport')
    return self.child.close()

  def read(self, n):
    data = self.child.read(n)
    hex_lines = self._to_hex(data)
    if len(hex_lines) > 1:
      self.logger.log(self.level, '%s read %4d B -> [%d B]:\n%s',
                      self.name, n, len(data), '\n'.join(hex_lines))
    else:
      self.logger.log(self.level, '%s read %4d B -> [%d B]: %s', self.name, n, len(data), hex_lines[0])

    return data

  def write(self, data):
    bytes_written = self.child.write(data)
    hex_lines = self._to_hex(data[:bytes_written])
    if len(hex_lines) > 1:
      self.logger.log(self.level, '%s write      <- [%d B]:\n%s', self.name, bytes_written, '\n'.join(hex_lines))
    else:
      self.logger.log(self.level, '%s write      <- [%d B]: %s', self.name, bytes_written, hex_lines[0])

    return bytes_written


class SerialTransport(Transport):

    _OPEN_PORTS = []

    @classmethod
    def close_atexit(cls):
        for port in cls._OPEN_PORTS:
            try:
                port.close()
            except Exception:
                _LOG.warn('exception closing port', exc_info=True)
                pass

        cls._OPEN_PORTS = []

    def __init__(self, grep=None, port_path=None, **kw):
        self._port_path = port_path
        self._grep = grep
        self._kw = kw
        if self._port_path is None and self._grep is None:
            raise SerialPortNotFoundError('Must specify one of grep= or port_path=')

    def open(self):
        if self._port_path is not None:
            port_path = self._port_path
        else:
            ports = list(serial.tools.list_ports.grep(self._grep, include_links=True))
            if len(ports) != 1:
                raise SerialPortNotFoundError(
                    f'grep expression should find 1 serial port; found {ports!r}')

            port_path = ports[0].device

        self._port = serial.Serial(port_path, timeout=0.1, exclusive=True, **self._kw)
        self._port.cancel_read()
        self._port.reset_input_buffer()
        self._port.reset_output_buffer()
        self._OPEN_PORTS.append(self._port)
#        self._port.read()
#        import time
#        time.sleep(1.1)

    def close(self):
        print(' *** PORT CLOSED! *** ')
        self._port.close()
        self._OPEN_PORTS.remove(self._port)
        self._port = None

    def read(self, n):
        to_return = bytearray()
        while not to_return:
            to_return.extend(self._port.read(n))

        while True:
            this_round = self._port.read(n - len(to_return))
            if not this_round:
                break
            to_return.extend(this_round)

        return to_return

    def write(self, data):
        # NOTE(areusch): Due to suspected flaky ST-Link VCP OS X driver, write 1 byte at a time.
        total_written = 0
        while len(data) > 0:
          num = self._port.write(data[:1])
          data = data[num:]
          assert num > 0
          total_written += num

        return total_written

atexit.register(SerialTransport.close_atexit)


class SubprocessTransport(Transport):

  def __init__(self, args, **kw):
    self.args = args
    self.kw = kw
    self.popen = None

  def open(self):
    self.kw['stdout'] = subprocess.PIPE
    self.kw['stdin'] = subprocess.PIPE
    self.kw['bufsize'] = 0
    self.popen = subprocess.Popen(self.args, **self.kw)
    self.stdin = self.popen.stdin
    self.stdout = self.popen.stdout

  def write(self, data):
    to_return = self.stdin.write(data)
    self.stdin.flush()

    return to_return

  def read(self, n):
    return self.stdout.read(n)

  def close(self):
    self.stdin.close()
    self.stdout.close()
    self.popen.terminate()


class DebugWrapperTransport(Transport):

  def __init__(self, debugger, transport):
    self.debugger = debugger
    self.transport = transport
    self.debugger.on_terminate_callbacks.append(self.transport.close)

  def open(self):
    self.debugger.Start()

    try:
      self.transport.open()
    except Exception:
      self.debugger.Stop()
      raise

  def write(self, data):
    return self.transport.write(data)

  def read(self, n):
    return self.transport.read(n)

  def close(self):
    self.transport.close()
    self.debugger.Stop()


TransportContextManager = typing.ContextManager[Transport]
