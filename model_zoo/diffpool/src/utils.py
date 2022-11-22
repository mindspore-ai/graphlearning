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
"""Diffpool utils"""
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.ops import functional as F
from mindspore.ops import operations as P

clip_grad = ms.ops.MultitypeFuncGraph("clip_grad")

@clip_grad.register("Number", "Tensor")
def _clip_grad(clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    dt = ops.dtype(grad)
    new_grad = nn.ClipByNorm()(grad, ops.cast(ops.tuple_to_array((clip_value,)), dt))
    return new_grad

class TrainOneStepCellWithGradClipping(ms.nn.TrainOneStepCell):
    """Train one step cell with grad clipping"""
    def __init__(self, net, optimizer, clip_val: float = 2.0) -> None:
        super().__init__(net, optimizer)
        self.clip = clip_val
        self.hyper_map = ops.HyperMap()

    def construct(self, *inputs):
        """construct function"""
        weights = self.weights
        loss = self.network(*inputs)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*inputs, sens)
        grads = self.hyper_map(F.partial(clip_grad, self.clip), grads)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        return loss
