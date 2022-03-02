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
"""reduction method for SharedNDArray"""
from multiprocessing.reduction import ForkingPickler
import mindspore_gl.memory_kernel as memory_kernel # pylint:disable=R0402
from .shared_memory import SharedMemory
from .shared_numpy import SharedNDArray


def rebuild_shared_ndarray(shm, shape, dtype) -> SharedNDArray:
    shm = SharedMemory(name=shm)
    shm_arr = SharedNDArray(shape, dtype=dtype, buffer=shm.array_buf)
    shm_arr.shm = shm
    shm_arr.shared = True
    return shm_arr


def reduce_shared_ndarray(arr: SharedNDArray):
    if arr.shared:
        raise Exception("An Already Shared Array Cannot Be Shared Again")
    memory_kernel.py_inc_ref(arr.shm.buf)

    return rebuild_shared_ndarray, (arr.shm.name, arr.shape, arr.dtype)


def init_reduction():
    ForkingPickler.register(SharedNDArray, reduce_shared_ndarray)
