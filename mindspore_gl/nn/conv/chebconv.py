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
"""Chebconv layer"""
import mindspore as ms
import mindspore.nn as nn
from mindspore._checkparam import Validator
from mindspore_gl import Graph
from mindspore_gl.nn import GNNCell

class ChebConv(GNNCell):
    r"""
    from the paper
    `Convolutional Neural Networks on Graphs with Fast Localized Spectral
    Filtering <https://arxiv.org/abs/1606.09375>`_ paper.

    .. math::
        \mathbf{X}^{\prime} = {\sigma}(\sum_{k=1}^{K} \mathbf{\beta}^{k} \cdot
        \mathbf{T}^{k} (\mathbf{\hat{L}}) \cdot X)

        \mathbf{\hat{L}} = 2 \mathbf{L} / {\lambda}_{max} - \mathbf{I}

    :math:`\mathbf{T}^{k}` is computed recursively by

    .. math::
        \mathbf{T}^{k}(\mathbf{\hat{L}}) = 2 \mathbf{\hat{L}}\mathbf{T}^{k-1}
        - \mathbf{T}^{k-2}

    where :math:`\mathbf{k}` is 1 or 2

    .. math::
        \mathbf{T}^{0} (\mathbf{\hat{L}}) = \mathbf{I}

        \mathbf{T}^{1} (\mathbf{\hat{L}}) = \mathbf{\hat{L}}

    Args:
        in_channels (int): Input node feature size.
        out_channels (int): Output node feature size.
        k (int): Chebyshev filter size. Default: 3
        bias (bool): Whether use bias. Default: True.

    Inputs:
        - **x** (Tensor) - The input node features. The shape is :math:`(N, D_{in})`
          where :math:`N` is the number of nodes,
          and :math:`D_{in}` should be equal to `in_channels` in `Args`.
        - **edge_weight** (Tensor) - Edge weights. The shape is :math:`(N\_e,)`
          where :math:`N\_e` is the number of edges.
        - **g** (Graph) - The input graph.

    Outputs:
        Tensor, output node features with shape of :math:`(N, D_{out})`, where :math:`(D_{out})` should be
        the same as `out_size` in `Args`.

    Raises:
        TypeError: If `in_channels` or `out_channels` or `k` is not an int.
        TypeError: If `bias` is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore_gl.nn import ChebConv
        >>> from mindspore_gl import GraphField
        >>> from mindspore_gl.utils import norm
        >>> n_nodes = 2
        >>> feat_size = 4
        >>> edge_index = [[0, 1], [1, 0]]
        >>> edge_index = ms.Tensor(edge_index, ms.int32)
        >>> ones = ms.ops.Ones()
        >>> feat = ones((n_nodes, feat_size), ms.float32)
        >>> edge_index, edge_weight = norm(edge_index, n_nodes)
        >>> feat = ones((n_nodes, feat_size), ms.float32)
        >>> checonv = ChebConv(in_channels=feat_size, out_channels=4, k=3)
        >>> res = checonv(feat, edge_weight, *graph_field.get_graph())
        >>> print(res.shape)
        (2, 4)
        """
    def __init__(self, in_channels: int, out_channels: int, k: int, bias: bool = True):
        super(ChebConv, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels, "in_channels", self.cls_name)
        self.out_channels = Validator.check_positive_int(out_channels, "out_channels", self.cls_name)
        self.k = Validator.check_positive_int(k, "k", self.cls_name)
        bias = Validator.check_bool(bias, "bias", self.cls_name)

        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lins = nn.CellList([nn.Dense(in_channels, out_channels, has_bias=True) for _ in range(k)])

        if bias:
            self.bias = ms.Parameter(ms.ops.Zeros()(self.out_channels, ms.float32))
        else:
            self.bias = None

    def construct(self, x, edge_weight, g: Graph):
        """
        Construct function for cheb layer.
        """
        cb_0 = x
        cb_1 = x
        out = self.lins[0](cb_0)
        if self.k > 1:
            g.set_vertex_attr({"x": x})
            for v in g.dst_vertex:
                feat = [u.x for u in v.innbs]
                v.x = g.sum(edge_weight * feat)
            cb_1 = [v.x for v in g.dst_vertex]
            out = out + self.lins[1](cb_1)
        for i in range(2, self.k):
            g.set_vertex_attr({"x": cb_1})
            for v in g.dst_vertex:
                feat = [u.x for u in v.innbs]
                v.x = g.sum(edge_weight * feat)
            cb_2 = [v.x for v in g.dst_vertex]
            cb_2 = 2. * cb_2 - cb_0
            out = out + self.lins[i](cb_2)
            cb_0, cb_1 = cb_1, cb_2
        if self.bias is not None:
            out += self.bias
        return out
        