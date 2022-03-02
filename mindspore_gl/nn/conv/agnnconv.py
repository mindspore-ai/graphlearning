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
"""AGNNConv Layer."""
import mindspore as ms
from mindspore_gl import Graph
from .. import GNNCell


class AGNNConv(GNNCell):
    r"""
    Attention Based Graph Neural Network.
    From the paper `Attention-based Graph Neural Network for Semi-Supervised Learning
    <https://arxiv.org/abs/1803.03735>`_.

    .. math::
        H^{l+1} = P H^{l}

    Computation of :math:`P` is:

    .. math::
        P_{ij} = \mathrm{softmax}_i ( \beta \cdot \cos(h_i^l, h_j^l))

    :math:`\beta` is a single scalar parameter.

    Args:
        init_beta (float): Init :math:`\beta`, a single scalar parameter.
        learn_beta (bool): Whether :math:`\beta` is learnable.
    """

    def __init__(self,
                 init_beta: float = 1.,
                 learn_beta: bool = True):
        super().__init__()
        if learn_beta:
            self.beta = ms.Parameter(ms.Tensor([init_beta], ms.float32))
        else:
            self.beta = ms.Tensor([init_beta], ms.float32)

    # pylint: disable=arguments-differ
    def construct(self, x, g: Graph):
        """
        Construct function for AGNNConv.

        Args:
            x (Tensor): The input node features.
            g (Graph): The input graph.

        Returns:
            Tensor, output node features.
        """
        g.set_vertex_attr({"h": x, "norm_h": ms.ops.L2Normalize()(x)})
        for v in g.dst_vertex:
            cosine_dis = [ms.ops.Exp()(self.beta * g.dot(u.norm_h, v.norm_h)) for u in v.innbs]
            # require axis=1 for the reduce_sum op, need explain to user
            a = cosine_dis / g.sum(cosine_dis)
            v.h = g.sum([u.h for u in v.innbs] * a)
        return [v.h for v in g.dst_vertex]
