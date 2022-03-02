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
"""GINConv Layer"""
import mindspore as ms
from mindspore_gl import Graph
from .. import GNNCell


class GINConv(GNNCell):
    r"""
    Graph isomorphic network layer.
    From the paper `How Powerful are Graph Neural Networks? <https://arxiv.org/pdf/1810.00826.pdf>`_.

    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \mathrm{aggregate}\left(\left\{h_j^{l}, j\in\mathcal{N}(i)
        \right\}\right)\right)

    If weights are provided on each edge, the weighted graph convolution is defined as:

    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \mathrm{aggregate}\left(\left\{e_{ji} h_j^{l}, j\in\mathcal{N}(i)
        \right\}\right)\right)

    Args:
        activation (Cell): Activation function.
        init_eps (int): Init value of eps.
        learn_eps (bool): Whether eps is learnable.
        aggregation_type (str): Type of aggregation, should in 'sum', 'max' and 'avg'.

    Raises:
        SyntaxError: when the aggregation type not in 'sum', 'max' and 'avg'.
    """

    def __init__(self,
                 activation: ms.nn.Cell,
                 init_eps=0.,
                 learn_eps=False,
                 aggregation_type="sum"):
        super().__init__()
        self.agg_type = aggregation_type
        if aggregation_type not in {"sum", "max", "avg"}:
            raise SyntaxError("Aggregation type must be one of sum, max or avg")
        if learn_eps:
            self.eps = ms.Parameter(ms.Tensor(init_eps, ms.float32))
        else:
            self.eps = ms.Tensor(init_eps, ms.float32)
        self.act = activation

    # pylint: disable=arguments-differ
    def construct(self, x, edge_weight, g: Graph):
        """
        Construct function for GINConv.

        Args:
            x (Tensor): The input node features.
            edge_weight (Tensor): Edge weights.
            g (Graph): The input graph.

        Returns:
            Tensor, output node features.
        """
        g.set_vertex_attr({"h": x})
        g.set_edge_attr({"w": edge_weight})
        for v in g.dst_vertex:
            if self.agg_type == 'sum':
                ret = g.sum([s.h * e.w for s, e in v.inedges])
            elif self.agg_type == 'max':
                ret = g.max([s.h * e.w for s, e in v.inedges])
            else:
                ret = g.avg([s.h * e.w for s, e in v.inedges])
            v.h = (1 + self.eps) * v.h + ret
            if self.act is not None:
                v.h = self.act(v.h)
        return [v.h for v in g.dst_vertex]
