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
"""utils"""

def get_indptr_from_coo_src(src_index: numpy.ndarray, result_array):
    """get indptr from coo"""
    cum_sum = 0
    index_ptr = 1
    cur_val = src_index[0]
    for index in range(src_index.shape[0]):
        if src_index[index] != cur_val:
            result_array[index_ptr] = cum_sum
            cur_val = src_index[index]
            index_ptr += 1
        else:
            cum_sum += 1
    return result_array
