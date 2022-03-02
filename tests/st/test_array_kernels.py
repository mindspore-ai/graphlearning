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
""" test array kernels """
import math
import numpy as np
import pytest
from mindspore_gl.graph.ops import PadArray2d, PadDirection, PadMode


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pad_const_local():
    """test pad const local"""
    input_array1 = np.ones([2, 140000])
    input_array2 = np.ones([2, 100000])
    pad_op = PadArray2d(direction=PadDirection.ROW, mode=PadMode.CONST, size=[2, 160000], dtype=input_array1.dtype,
                        fill_value=30)
    res = pad_op(input_array1)
    assert res.shape[1] == 160000

    res = pad_op(input_array2)
    assert res.shape[1] == 160000

    assert res[0][-1] == 30


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pad_auto_local():
    """test pad auto local"""
    input_array1 = np.ones([2, 140000])
    pad_op = PadArray2d(direction=PadDirection.ROW, mode=PadMode.AUTO, dtype=input_array1.dtype, fill_value=30)
    res = pad_op(input_array1)
    assert res.shape[1] == (1 << (math.ceil(math.log2(140000))))

    input_array2 = np.ones([2, 3200])
    res = pad_op(input_array2)
    assert res.shape[1] == 4096
    assert res[0][-1] == 30

    pad_op = PadArray2d(direction=PadDirection.ROW, mode=PadMode.AUTO, dtype=input_array1.dtype)
    res = pad_op(input_array2)

    assert res.shape[1] == 4096
    assert res[0][-1] == 4095


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pad_const_shared():
    """test pad const shared"""
    input_array1 = np.ones([2, 140000])
    pad_op = PadArray2d(direction=PadDirection.ROW, mode=PadMode.CONST, size=[2, 160000], dtype=input_array1.dtype,
                        fill_value=30, use_shared_numpy=True)
    res = pad_op(input_array1)
    assert res.shape[1] == 160000

    input_array2 = np.ones([2, 100000])
    res = pad_op(input_array2)
    assert res.shape[1] == 160000
    assert res[0][-1] == 30


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pad_auto_shared():
    """test pad auto shared"""
    input_array1 = np.ones([2, 140000])
    pad_op = PadArray2d(direction=PadDirection.ROW, mode=PadMode.AUTO, dtype=input_array1.dtype,
                        fill_value=30, use_shared_numpy=True)
    res = pad_op(input_array1)
    assert res.shape[1] == (1 << (math.ceil(math.log2(140000))))

    input_array2 = np.ones([2, 3200])
    res = pad_op(input_array2)
    assert res.shape[1] == 4096
    assert res[0][-1] == 30

    pad_op = PadArray2d(direction=PadDirection.ROW, mode=PadMode.AUTO, dtype=input_array1.dtype)
    res = pad_op(input_array2)

    assert res.shape[1] == 4096
    assert res[0][-1] == 4095
