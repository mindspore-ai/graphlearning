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
""" test shared numpy """
import numpy as np
import mindspore as ms
import mindspore_gl.dataloader.shared_numpy as shared_numpy
import pytest


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_queue():
    """
    Feature: use `Queue` manage multiprocess data.
    Description: None.
    Expectation: success.
    """
    queue = shared_numpy.Queue()
    arr = np.ones([10000, 600], dtype=np.float32)
    queue.put(arr)
    ret = queue.get()
    assert ret.shape[0] == 10000
    assert ret.shape[1] == 600


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_create_from_numpy():
    """
    Feature: construct `SharedNDArray` from numpy array.
    Description: numpy_array with shape [100000, 600].
    Expectation: success.
    """
    numpy_array = np.ones([100000, 600], dtype=np.float32)
    shared_array = shared_numpy.SharedNDArray.from_numpy_array(numpy_array)
    shared_array[0, 10:30] = 1
    assert all(shared_array[0, 10:30] == 1)
    ms.Tensor.from_numpy(shared_array)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_create_from_shape():
    """
        Feature: construct `SharedNDArray` with array size and data type.
        Description: target_size = [10000, 600] , dtype = np.int32
        Expectation: success.
        """
    target_size = [10000, 600]
    shared_array = shared_numpy.SharedNDArray.from_shape(target_size, dtype=np.int32)
    shared_array[0, 10:30] = 1
    assert all(shared_array[0, 10:30] == 1)
    ms.Tensor.from_numpy(shared_array)
