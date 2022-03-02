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
"""Graph Utils."""
from typing import Union, List, Tuple, Iterable
import numpy as np
import mindspore_gl.dataloader.shared_numpy as shared_numpy
import mindspore_gl.memory_kernel as memory_kernel


class ArrayPool:
    """
    Memory pool for reuse
    """
    def __init__(self):
        self.array_pool = {}

    def put(self, size, array: np.ndarray):
        """
        put a array into array pool

        Args:
            size(Union[List, Tuple]): input array size
            array(numpy.array): input array
        """
        key = "_".join(list(map(str, size)))
        if self.array_pool.get(key, None) is None:
            self.array_pool[key] = []
        self.array_pool[key].append(array)

    def pop(self, size) -> np.ndarray:
        """
        pop a array from array pool

        Args:
            size(Union[List, Tuple]): request array's size

        Returns:
            Union[numpy.array, None], return None is request size has no array left, else return the array.

        """
        key = "_".join(list(map(str, size)))
        buckets = self.array_pool.get(key, None)
        if buckets is None:
            return None

        return buckets.pop()


class SharedArrayPool:
    """
    Shared memory pool for reuse, this is recommended for interprocess communication
    """
    def __init__(self):
        self.array_pool = {}

    def put(self, size: Union[List, Tuple, Iterable], shared_array: shared_numpy.SharedNDArray):
        """
         put a array into array pool

        Args:
            size(Union[List, Tuple]): input array size
            shared_array(shared_numpy.SharedNDArray): input array
        """
        key = "_".join(list(map(str, size)))
        if self.array_pool.get(key, None) is None:
            self.array_pool[key] = []
        self.array_pool[key].append(shared_array)

    def check_avaliable(self, shared_array: shared_numpy.SharedNDArray):
        """
        check if this memory can be reused

        Args:
            shared_array: input shared_numpy.SharedNDArray

        Returns:
            bool, indicate if certain shared_array if available for reuse.
        """
        return memory_kernel.py_ref_count(shared_array.shm.buf) == 1 and not shared_array.shared

    def pop(self, size: Union[List, Tuple, Iterable]) -> shared_numpy.SharedNDArray:
        """
        pop a array from array pool

        Args:
            size(Union[List, Tuple]): request array's size

        Returns:
            Union[numpy.array, None], return None is request size has no array left, else return the array.

        """
        key = "_".join(list(map(str, size)))
        bucket = self.array_pool.get(key, None)
        if bucket is None:
            return None
        # pylint: disable=consider-using-enumerate
        for idx in range(len(bucket)):
            if self.check_avaliable(bucket[idx]):
                return bucket.pop(idx)
        return None
