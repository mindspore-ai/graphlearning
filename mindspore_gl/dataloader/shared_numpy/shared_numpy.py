"""SharedNDArray."""
from typing import List, Tuple, Union, Iterable
import numpy as np
import mindspore_gl.memory_kernel as memory_kernel # pylint:disable=R0402

from .shared_memory import SharedMemory


__all__ = ["SharedNDArray"]


def prod(shape: Union[List[int], Tuple, Iterable]):
    res = 1
    for item in shape:
        res *= item
    return res


class SharedNDArray(np.ndarray):
    """
    SharedNDArray, memory is allocated in shm. It's recommended for inter-process communication.
    """

    @property
    def shm(self) -> SharedMemory:
        return self._shm

    @shm.setter
    def shm(self, shm):
        self._shm = shm

    @property
    def shared(self) -> bool:
        if not hasattr(self, "_shared"):
            self._shared = False
        return self._shared

    @shared.setter
    def shared(self, shared):
        self._shared = shared

    def close(self):
        self._shm.close()

    def unlink(self):
        self._shm.unlink()

    def __del__(self):
        if hasattr(self, "_shm"):
            ##################
            # Decr Ref Count
            ##################
            memory_kernel.py_dec_ref(self.shm.buf)
            ref_count = memory_kernel.py_ref_count(self.shm.buf)

            ##################################################
            # Unlink Memory When No One Reference This Memory
            ##################################################
            if ref_count > 0:
                self.close()
            else:
                self.unlink()

    @classmethod
    def from_numpy_array(cls, arr: np.ndarray):
        """
        Create SharedNDArrray from numpy.array.

        Args:
            arr(numpy.array): numpy array will be created as shared memory block.

        Outputs:
            SharedNDArray, SharedNDArray created from numpy array.

        Examples:
            >>> import numpy as np
            >>> from mindspore_gl.dataloader import SharedNDArray
            >>> np_arr = np.zeros([1000, 500], dtype=np.float32)
            >>> shared_arr = SharedNDArray.from_numpy_array(np_arr)
            >>> shared_arr[0, 0:5] = 2
            >>> print(shared_arr[0:2, 0:5])
                [[2, 2, 2, 2, 2], [1, 1, 1, 1, 1]]
        """
        ############################
        # Allocate SharedMemory
        ############################
        shm = SharedMemory(create=True, size=arr.nbytes)
        shm_arr = SharedNDArray(arr.shape, dtype=arr.dtype, buffer=shm.array_buf)
        shm_arr[:] = arr[:]

        #######################################
        # Set Attributes, Init Memory RefCount
        #######################################
        shm_arr.shm = shm
        shm_arr.shared = False
        memory_kernel.py_init_refcount(shm.buf)

        return shm_arr

    @classmethod
    def from_shape(cls, shape: Union[List[int], Tuple, Iterable], dtype: np.dtype):
        """
        Create SharedNDArray from shape and dtype.

        Args:
            shape(List): array shape of the shared array.
            dtype(numpy.dtype): data type of the shared array.

        Outputs:
            SharedNDArray, array created.

        Examples:
            >>> from mindspore_gl.dataloader import SharedNDArray
            >>> import numpy as np
            >>> tgt_size = [1000, 500]
            >>> shared_arr = SharedNDArray.from_shape(tgt_size, dtype=np.int32)
            >>> shared_arr[0, 0:5] = 1
            >>> print(shared_arr[0, 0:5])
                [[1, 1, 1, 1, 1]]

        """
        if not isinstance(shape, list):
            raise TypeError("'shape' should be list, but got {}.".format(shape))

        dtype = np.dtype(dtype)
        array_size = prod(shape) * dtype.itemsize
        shm = SharedMemory(create=True, size=array_size)
        shm_arr = SharedNDArray(shape, dtype=dtype, buffer=shm.array_buf)
        #######################################
        # Set Attributes, Init Memory RefCount
        #######################################
        shm_arr.shm = shm
        shm_arr.shared = False
        memory_kernel.py_init_refcount(shm.buf)
        return shm_arr
