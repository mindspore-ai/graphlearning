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
"""EGConv Layer"""
from typing import List

import mindspore as ms
from mindspore.common.initializer import initializer
from mindspore.common.initializer import XavierUniform
from mindspore._checkparam import Validator
from mindspore_gl import Graph
from .. import GNNCell


class EGConv(GNNCell):
    r"""
    Efficient Graph Convolution from the paper `Adaptive Filters and Aggregator Fusion for Efficient Graph Convolutions
    <https://arxiv.org/abs/2104.01481>`_.

    .. math::
        h_i^{(l+1)} = {\LARGE ||}_{h=1}^{H} \sum_{\oplus \in \mathcal{A}} \sum_{b=1}^{B} w_{h,\oplus,b}^{(l)}
        \bigoplus_{j \in \mathcal{N(i)}} W_{b}^{(l)} h_{j}^{(l)}

    :math:`\mathcal{N}(i)` represents the neighbour node of :math:`i`,
    :math:`W_{b}^{(l)}` represents a basis weight,
    :math:`\oplus` represents an aggregator,
    :math:`w_{h,\oplus,b}^{(l)}` represents per-vertex weighting coefficients across heads, aggregator and bases.

    Args:
        in_feat_size (int): Input node feature size.
        out_feat_size (int): Output node feature size.
        aggregators (List[str]): aggregators to be used. Supported aggregators are
            `sum`, `mean`, `max`, `min`, `std`, `var`, `symnorm`. Default: 'symnorm'.
        num_heads (int, optional): Number of heads :math:`H`. Default: 8. Must have `out_feat_size % num_heads == 0`.
        num_bases (int, optional): Number of basis weight :math:`B`. Default: 4.
        bias (bool, optional): Whether the layer will learn an additive bias. Default: True.

    Inputs:
        - **x** (Tensor) - The input node features. The shape is :math:`(N, D_{in})`
          where :math:`N` is the number of nodes,
          and :math:`D_{in}` should be equal to `in_feat_size` in `Args`.
        - **g** (Graph) - The input graph.

    Outputs:
        - Tensor, output node features with shape of :math:`(N, D_{out})`, where :math:`(D_{out})` should be the same as
          `out_feat_size` in `Args`.

    Raises:
        TypeError: If `in_feat_size` or `out_feat_size` or `num_heads` is not a positive int.
        ValueError: If `out_feat_size` is not divisible by 'num_heads'.
        ValueError: If `aggregators` is not in ['sum', 'mean', 'max', 'min', 'symnorm', 'var', 'std'].

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore_gl.nn import EGConv
        >>> from mindspore_gl import GraphField
        >>> n_nodes = 4
        >>> n_edges = 7
        >>> feat_size = 4
        >>> src_idx = ms.Tensor([0, 1, 1, 2, 2, 3, 3], ms.int32)
        >>> dst_idx = ms.Tensor([0, 0, 2, 1, 3, 0, 1], ms.int32)
        >>> ones = ms.ops.Ones()
        >>> feat = ones((n_nodes, feat_size), ms.float32)
        >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
        >>> conv = EGConv(in_feat_size=4, out_feat_size=6, aggregators=['sum'], num_heads=3, num_bases=3)
        >>> res = conv(feat, *graph_field.get_graph())
        >>> print(res.shape)
        (4, 6)
    """
    def __init__(self,
                 in_feat_size: int,
                 out_feat_size: int,
                 aggregators: List[str],
                 num_heads: int = 8,
                 num_bases: int = 4,
                 bias: bool = True
                 ) -> None:
        super().__init__()
        self.in_feat_size = Validator.check_positive_int(in_feat_size, "in_feat_size", self.cls_name)
        self.out_feat_size = Validator.check_positive_int(out_feat_size, "out_feat_size", self.cls_name)
        self.num_heads = Validator.check_positive_int(num_heads, "num_heads", self.cls_name)

        if out_feat_size % num_heads != 0:
            raise ValueError(f"For '{self.cls_name}', the 'out_feat_size' should be divisible by 'num_heads', "
                             f"but got out_feat_size: {out_feat_size}, num_heads: {num_heads}.")
        self.num_bases = num_bases
        for agg in aggregators:
            if agg not in ['sum', 'mean', 'max', 'min', 'symnorm', 'var', 'std']:
                raise ValueError(f"For '{self.cls_name}', the aggregator: '{agg}' is unsupported.")
        self.agg_num = len(aggregators)
        self.aggregators = aggregators
        self.basis_fc = ms.nn.Dense(in_feat_size, (out_feat_size // num_heads) * num_bases,
                                    weight_init=XavierUniform(), has_bias=False)
        self.combine_fc = ms.nn.Dense(in_feat_size, num_heads * num_bases * self.agg_num,
                                      weight_init=XavierUniform(), has_bias=True)
        if bias:
            self.bias = ms.Parameter(initializer('zero', (out_feat_size), ms.float32), name="bias")
        else:
            self.bias = None
        self.reshape = ms.ops.Reshape()
        self.matmul = ms.nn.MatMul()
        self.sqrt = ms.ops.Sqrt()
        self.relu = ms.ops.ReLU()
        self.stack = ms.ops.Stack(axis=1)
        self.eps = 1e-5

    def combine(self, weights, aggregated):
        aggregated = aggregated.view(-1, self.agg_num * self.num_bases, self.out_feat_size // self.num_heads)
        x = self.matmul(weights, aggregated)
        x = x.view(-1, self.out_feat_size)
        if self.bias is not None:
            x = x + self.bias
        return x

    # pylint: disable=arguments-differ
    def construct(self, x, g: Graph):
        """
        Construct function for EGConv.
        """
        bases = self.basis_fc(x)
        weights = self.combine_fc(x)
        weights = self.reshape(weights, (-1, self.num_heads, self.num_bases * self.agg_num))

        outs = []
        for agg in self.aggregators:
            if agg == 'symnorm':
                in_deg = self.sqrt(1.0 / g.in_degree())
                g.set_vertex_attr({'x': bases, 'deg': in_deg})
                for v in g.dst_vertex:
                    v.x = g.sum([u.deg * u.x for u in v.innbs])
                out = [v.x for v in g.dst_vertex] * in_deg
            elif agg in ['var', 'std']:
                g.set_vertex_attr({'x': bases, 'x_square': bases * bases})
                for v in g.dst_vertex:
                    v.x = g.avg([u.x for u in v.innbs])
                    v.x_square = g.avg([u.x_square for u in v.innbs])
                out = [v.x_square - v.x * v.x for v in g.dst_vertex]
                if agg == 'std':
                    out = self.sqrt(self.relu(out) + self.eps)
            else:
                g.set_vertex_attr({'x': bases})
                for v in g.dst_vertex:
                    x_list = [u.x for u in v.innbs]
                    if agg == 'sum':
                        v.x = g.sum(x_list)
                    elif agg == 'mean':
                        v.x = g.avg(x_list)
                    elif agg == 'max':
                        v.x = g.max(x_list)
                    elif agg == 'min':
                        v.x = g.min(x_list)
                out = [v.x for v in g.dst_vertex]
            outs.append(out)
        aggregated = self.stack(outs)
        return self.combine(weights, aggregated)
