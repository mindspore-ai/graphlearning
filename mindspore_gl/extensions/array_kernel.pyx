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

import numpy as np
cimport numpy as np
cimport cython
from libcpp.map cimport map
from libcpp.set cimport set
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libc.stdlib cimport rand, RAND_MAX
from libcpp cimport bool
from cython.parallel import prange

@cython.boundscheck(False)
@cython.boundscheck(False)
def float_2d_array_slicing(np.ndarray[np.float32_t, ndim=2] array, np.ndarray[np.int32_t, ndim=1] indices):

    cdef np.ndarray[np.float32_t, ndim=2] res = np.zeros([indices.shape[0], array.shape[1]], dtype=array.dtype)
    cdef int indice_size =indices.shape[0]
    cdef int index
    cdef int indice_value

    cdef int [:] indice_view = indices
    cdef float [:, :] array_view = array
    cdef float [:, :] res_view = res
    with nogil:
        for index in prange(indice_size, schedule="static"):
            indice_value = indice_view[index]
            res_view[index] = array_view[indice_value]
    return res

@cython.boundscheck(False)
@cython.boundscheck(False)
def int_2d_array_slicing(np.ndarray[np.int32_t, ndim=2] array, np.ndarray[np.int32_t, ndim=1] indices):
    cdef np.ndarray[np.int32_t, ndim=2] res = np.zeros([indices.shape[0], array.shape[1]], dtype=array.dtype)
    cdef int indice_size = indices.shape[0]
    cdef int index
    cdef int indice_value
    for index in xrange(indice_size):
        indice_value = indices[index]
        res[index] = array[indice_value]
    return res

@cython.boundscheck(False)
@cython.boundscheck(False)
def int_1d_array_slicing(np.ndarray[np.int32_t, ndim=1] array, np.ndarray[np.int32_t, ndim=1] indices):
    cdef np.ndarray[np.int32_t, ndim=1] res = np.zeros([indices.shape[0]], dtype=array.dtype)
    cdef int indice_size = indices.shape[0]
    cdef int index
    cdef int indice_value
    for index in xrange(indice_size):
        indice_value = indices[index]
        res[index] = array[indice_value]
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
def float_2d_array_col_copy(np.ndarray[np.float32_t, ndim=2] target_array, np.ndarray[np.float32_t, ndim=2] source_array):
    cdef float [:, :] target_view = target_array
    cdef float [:, :] source_view = source_array
    cdef Py_ssize_t total_count = source_array.shape[0]
    cdef int length = source_array.shape[1]
    cdef Py_ssize_t index
    cdef int row

    with nogil:
        for row in prange(total_count, schedule="static"):
            target_view[row, :] = source_view[row, :]
    return target_array

@cython.boundscheck(False)
@cython.boundscheck(False)
def float_2d_gather_with_dst(np.ndarray[np.float32_t, ndim=2] dst,
                             np.ndarray[np.float32_t, ndim=2] src,
                             np.ndarray[np.int32_t, ndim=1] indices
                            ):

    cdef int indice_size =indices.shape[0]
    cdef int index
    cdef int indice_value

    cdef int [:] indice_view = indices
    cdef float [:, :] dst_view = dst
    cdef float [:, :] src_view = src
    with nogil:
        for index in prange(indice_size, schedule="static"):
            indice_value = indice_view[index]
            dst_view[index] = src_view[indice_value]
    return dst