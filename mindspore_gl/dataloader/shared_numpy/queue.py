# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Queue for SharedNDArray."""
import io
import pickle
import multiprocessing
import multiprocessing.queues

from multiprocessing.reduction import ForkingPickler


class ConnectionWrapper:
    """
    Proxy class for _multiprocessing.Connection which uses ForkingPickler to
    serialize objects.
    """

    def __init__(self, conn):
        self.conn = conn

    def send(self, obj):
        buf = io.BytesIO()
        ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(obj)
        self.send_bytes(buf.getvalue())

    def recv(self):
        buf = self.recv_bytes()
        return pickle.loads(buf)

    def __getattr__(self, name):
        if "conn" in self.__dict__:
            return getattr(self.conn, name) # pylint:disable=C0209
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, "conn"))# pylint:disable=C0209


class Queue(multiprocessing.queues.Queue):
    """
    MultiProcessing Queue for inter-process communication.

    Examples:
        >>> from mindspore_gl.dataloader.shared_numpy import queue
        >>> import numpy as np
        >>> array = np.ones([2500, 602], dtype=np.float32)
        >>> queue.put(array)
        >>> ret = queue.get()
        >>> print(ret.shape)
            2500, 602


    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, ctx=multiprocessing.get_context(), **kwargs)
        self._reader = ConnectionWrapper(self._reader)
        self._writer = ConnectionWrapper(self._writer)
        self._send = self._writer.send
        self._recv = self._reader.recv
