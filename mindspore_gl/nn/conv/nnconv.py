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
"""NNConv Layer"""
import math

import mindspore as ms
from mindspore.common.initializer import XavierUniform
from mindspore_gl import Graph
from .. import GNNCell


class NNConv(GNNCell):
    r"""
    Graph convolutional layer.
    From the paper `Neural Message Passing for Quantum Chemistry <https://arxiv.org/pdf/1704.01212.pdf>`_.

    .. math::
        h_{i}^{l+1} = h_{i}^{l} + \mathrm{aggregate}\left(\left\{
        f_\Theta (e_{ij}) \cdot h_j^{l}, j\in \mathcal{N}(i) \right\}\right)

    Where :math:`f_\Theta` is a function with learnable parameters.

    Args:
        in_feat_size (int): Input node feature size.
        out_feat_size (int): Output node feature size.
        edge_embed (Cell): Edge embedding function Cell.
        aggregator_type (str): Type of aggregator, should be 'sum'.
        residual (bool): Whether use residual.
        bias (bool): Whether use bias.
    """

    def __init__(self,
                 in_feat_size: int,
                 out_feat_size: int,
                 edge_embed: ms.nn.Cell,
                 aggregator_type: str = "sum",
                 residual=False,
                 bias=True):
        super().__init__()
        if aggregator_type != "sum":
            raise TypeError("Don't support aggregator type other than sum.")
        self.edge_embed = edge_embed
        self.agg_type = aggregator_type
        self.in_feat_size = in_feat_size
        self.out_feat_size = out_feat_size
        self.res_dense = None
        if residual:
            if in_feat_size != out_feat_size:
                self.res_dense = ms.nn.Dense(in_feat_size, out_feat_size, weight_init=XavierUniform(math.sqrt(2)))
        self.bias = None
        if bias:
            self.bias = ms.Parameter(ms.ops.Zeros()((out_feat_size), ms.float32))

    # pylint: disable=arguments-differ
    def construct(self, x, edge_feat, g: Graph):
        """
        Construct function for NNConv.

        Args:
            x (Tensor): The input node features.
            edge_feat (Tensor): The input edge features.
            g (Graph): The input graph.

        Returns:
            Tensor, output node features.
        """
        g.set_vertex_attr({"h": ms.ops.ExpandDims()(x, -1)})
        g.set_edge_attr(
            {"g": ms.ops.Reshape()(self.edge_embed(edge_feat), (-1, self.in_feat_size, self.out_feat_size))})
        for v in g.dst_vertex:
            e = [s.h * e.g for s, e in v.inedges]
            v.rt = g.sum(e)
            v.rt = ms.ops.ReduceSum()(v.rt, 1)
            if self.res_dense is not None:
                v.rt = v.rt + self.res_dense(v.h)
            if self.bias is not None:
                v.rt = v.rt + self.bias
        return [v.rt for v in g.dst_vertex]
