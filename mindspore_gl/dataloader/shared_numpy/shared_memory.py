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
"""
Provides shared memory for direct access across processes.
The API of this package is currently provisional. Refer to the
documentation for details.
"""

__all__ = ['SharedMemory']

import mmap
import os
import secrets

import mindspore_gl.dataloader.shared_numpy._posixshmem as _posixshmem  # pylint:disable=R0402

_O_CREX = os.O_CREAT | os.O_EXCL

# FreeBSD (and perhaps other BSDs) limit names to 14 characters.
_SHM_SAFE_NAME_LENGTH = 14

# Shared memory block name prefix
_SHM_NAME_PREFIX = 'psm_'


def _make_filename():
    "Create a random filename for the shared memory object."
    # number of random bytes to use for name
    nbytes = (_SHM_SAFE_NAME_LENGTH - len(_SHM_NAME_PREFIX)) // 2
    if nbytes < 2:
        raise ValueError('_SHM_NAME_PREFIX too long')
    name = _SHM_NAME_PREFIX + secrets.token_hex(nbytes)
    if len(name) > _SHM_SAFE_NAME_LENGTH:
        raise ValueError('_SHM_NAME_PREFIX too long')
    return name


class SharedMemory:
    """
    Creates a new shared memory block or attaches to an existing
    shared memory block.

    Every shared memory block is assigned a unique name.  This enables
    one process to create a shared memory block with a particular name
    so that a different process can attach to that same shared memory
    block using that same name.

    As a resource for sharing data across processes, shared memory blocks
    may outlive the original process that created them.  When one process
    no longer needs access to a shared memory block that might still be
    needed by other processes, the close() method should be called.
    When a shared memory block is no longer needed by any process, the
    unlink() method should be called to ensure proper cleanup.

    Args:
        name(str or None): unique name for shared memory block.
        create(bool): set True, create a new shared memory block; set False, attach to a existing shared memory block.
        size(int): specifies the requested number of bytes when creating a new shared memory block.
            When attaching to an existing shared memory block, the size parameter is ignored.

    Examples:
        >>> from mindspore_gl.dataloader.shared_numpy import SharedMemory
        >>> import numpy as np
        >>> array = np.ones([2500, 602], dtype=np.float32)
        >>> shm = SharedMemory(create=True, size=array.nbytes)

    """

    # Defaults; enables close() and unlink() to run without errors.
    _name = None
    _fd = -1
    _mmap = None
    _buf = None
    _flags = os.O_RDWR
    _map_alloc_alignment = 64
    _mode = 0o600

    def __init__(self, name=None, create=False, size=0):
        if not isinstance(size, int) or size < 0:
            raise TypeError("'size' must be a positive integer, but got {}.".format(size))
        if not isinstance(create, bool):
            raise TypeError("'create' must be a bool, but got {}.".format(create))
        if create:
            self._flags = _O_CREX | os.O_RDWR
        if name is None and not self._flags & os.O_EXCL:
            raise ValueError("'name' can only be None if create=True")

        if name is None:
            while True:
                name = _make_filename()
                try:
                    self._fd = _posixshmem.shm_open(
                        name,
                        self._flags,
                        mode=self._mode
                    )

                except FileExistsError:
                    continue
                self._name = name
                break
        else:
            if not isinstance(name, str):
                raise TypeError("'name' must be str, but got {}.".format(name))
            self._fd = _posixshmem.shm_open(
                name,
                self._flags,
                mode=self._mode
            )

            self._name = name
        try:
            if create and size:
                os.ftruncate(self._fd, size + self._map_alloc_alignment)
            stats = os.fstat(self._fd)
            size = stats.st_size
            self._mmap = mmap.mmap(self._fd, size)
        except OSError:
            self.unlink()
            raise

        self._size = size
        self._buf = memoryview(self._mmap)

    def __del__(self):
        try:
            self.close()
        except OSError:
            pass

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.name,
                False,
                self.size,
            ),
        )

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name!r}, size={self.size})'

    @property
    def buf(self):
        """
        A memoryview of contents of the shared memory block.
        """
        return self._buf

    @property
    def array_buf(self):
        return self._buf[self._map_alloc_alignment:]

    @property
    def name(self):
        """
        Unique name that identifies the shared memory block.
        """
        return self._name

    @property
    def size(self):
        """
        Size in bytes.
        """
        return self._size

    def close(self):
        """
        Closes access to the shared memory from this instance but does
        not destroy the shared memory block.
        """
        if self._buf is not None:
            self._buf.release()
            self._buf = None
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def unlink(self):
        """
        Requests that the underlying shared memory block be destroyed.

        In order to ensure proper cleanup of resources, unlink should be
        called once (and only once) across all processes which have access
        to the shared memory block.
        """
        if self.name:
            _posixshmem.shm_unlink(self.name)
