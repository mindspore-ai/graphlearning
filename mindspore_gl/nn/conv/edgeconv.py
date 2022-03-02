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
"""EDGEConv Layer"""
import mindspore as ms
from mindspore_gl import Graph
from .. import GNNCell


class EDGEConv(GNNCell):
    r"""
    EdgeConv layer, from the paper `Dynamic Graph CNN for Learning on Point Clouds <https://arxiv.org/pdf/1801.07829>`_.

    .. math::
        h_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} (
       \Theta \cdot (h_j^{(l)} - h_i^{(l)}) + \Phi \cdot h_i^{(l)})

    :math:`\mathcal{N}(i)` represents the neighbour node of :math:`i`.
    :math:`\Theta` and :math:`\Phi` represents linear layers.

    Args:
        in_feat_size (int): Input node feature size.
        out_feat_size (int): Output node feature size.
        batch_norm (bool): Whether use batch norm.
        bias (bool): Whether use bias.
    """

    # pylint: disable=arguments-differ
    def __init__(self,
                 in_feat_size: int,
                 out_feat_size: int,
                 batch_norm: bool,
                 bias=True):
        super().__init__()
        self.batch_norm = batch_norm
        self.theta = ms.nn.Dense(in_feat_size, out_feat_size, has_bias=bias)
        self.phi = ms.nn.Dense(in_feat_size, out_feat_size, has_bias=bias)
        if batch_norm:
            self.bn = ms.nn.BatchNorm1d(out_feat_size)

    def construct(self, x, g: Graph):
        """
        Construct function for EDGEConv.

        Args:
            x (Tensor): The input node features.
            g (Graph): The input graph.

        Returns:
            Tensor, output node features.
        """
        g.set_vertex_attr({"x": x, "phi": self.phi(x)})
        for v in g.dst_vertex:
            if not self.batch_norm:
                v.h = g.max([self.theta(u.x - v.x) + v.phi for u in v.innbs])
            else:
                v.h = g.max([self.bn(self.theta(u.x - v.x) + v.phi) for u in v.innbs])
        return [v.h for v in g.dst_vertex]
