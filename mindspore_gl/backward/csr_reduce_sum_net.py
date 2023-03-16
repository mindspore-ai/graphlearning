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
"""User-defined csr_reduce_sum backwards"""
import mindspore as ms


class CSRReduceSumNet(ms.nn.Cell):
    """
    Redefine back-propagation to make use of sorted indices.
    """

    # pylint: disable=W0212
    def __init__(self):
        super().__init__()
        self.op = ms.ops.operations._csr_ops.CSRReduceSum()

    # pylint: disable=W0613
    def construct(self, indptr, indices, values, shape, axis, indices_backward):
        return self.op(indptr, indices, values, shape, axis)

    # pylint: disable=W0613
    def bprop(self, indptr, indices, values, shape, axis, indices_backward, out=0, dout=0):
        dout = dout.reshape((dout.shape[0],) + dout.shape[2:])
        grad_values = ms.ops.gather(dout, indices_backward, 0)
        return indptr, indices, grad_values, shape, axis, indices_backward
