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
from mindspore import nn
from mindspore._checkparam import Validator
from mindspore_gl import Graph
from .. import GNNCell


class NNConv(GNNCell):
    r"""
    Graph convolutional layer.
    From the paper `Neural Message Passing for Quantum Chemistry <https://arxiv.org/pdf/1704.01212.pdf>`_ .

    .. math::
        h_{i}^{l+1} = h_{i}^{l} + \mathrm{aggregate}\left(\left\{
        f_\Theta (e_{ij}) \cdot h_j^{l}, j\in \mathcal{N}(i) \right\}\right)

    Where :math:`f_\Theta` is a function with learnable parameters.

    Args:
        in_feat_size (int): Input node feature size.
        out_feat_size (int): Output node feature size.
        edge_embed (Cell): Edge embedding function Cell.
        aggregator_type (str): Type of aggregator, should be 'sum'. Default: sum.
        residual (bool): Whether use residual. Default: False.
        bias (bool): Whether use bias. Default: True.

    Inputs:
        - **x** (Tensor) - The input node features. The shape is :math:`(N,D\_in)`
          where :math:`N` is the number of nodes and :math:`D\_in` could be of any shape.
        - **edge_feat** (Tensor) - Edge featutes. The shape is :math:`(N\_e,F\_e)`
          where :math:`N\_e` is the number of edges and :math:`F\_e` is the number of edge features.
        - **g** (Graph) - The input graph.

    Outputs:
        - Tensor, the output feature of shape :math:`(N,D\_out)`
          where :math:`N` is the number of nodes and :math:`D\_out` could be of any shape.

    Raises:
        TypeError: if `edge_embed` type is not mindspore.nn.Cell or `aggregator_type` is not sum
        TypeError: If `in_feat_size` or `out_feat_size` is not an int.
        TypeError: If `residual` or `bias` is not a bool.

    Supported Platforms:
        ``GPU`` ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore_gl.nn import NNConv
        >>> from mindspore_gl import GraphField
        >>> n_nodes = 4
        >>> n_edges = 7
        >>> node_feat_size = 7
        >>> edge_feat_size = 4
        >>> src_idx = ms.Tensor([0, 1, 1, 2, 2, 3, 3], ms.int32)
        >>> dst_idx = ms.Tensor([0, 0, 2, 1, 3, 0, 1], ms.int32)
        >>> ones = ms.ops.Ones()
        >>> node_feat = ones((n_nodes, node_feat_size), ms.float32)
        >>> edge_feat = ones((n_edges, edge_feat_size), ms.float32)
        >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
        >>> edge_func = ms.nn.Dense(edge_feat_size, 2)
        >>> nnconv = NNConv(in_feat_size=node_feat_size, out_feat_size=2, edge_embed=edge_func)
        >>> res = nnconv(node_feat, edge_feat, *graph_field.get_graph())
        >>> print(res.shape)
        (4, 2)
    """

    def __init__(self,
                 in_feat_size: int,
                 out_feat_size: int,
                 edge_embed: ms.nn.Cell,
                 aggregator_type: str = "sum",
                 residual=False,
                 bias=True):
        super().__init__()
        in_feat_size = Validator.check_positive_int(in_feat_size, "in_feat_size", self.cls_name)
        out_feat_size = Validator.check_positive_int(out_feat_size, "out_feat_size", self.cls_name)
        residual = Validator.check_bool(residual, "bias", self.cls_name)
        bias = Validator.check_bool(bias, "bias", self.cls_name)
        if edge_embed:
            if not isinstance(edge_embed, nn.Cell):
                raise TypeError("edge_embed type should be ms.nn.Cell")
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
