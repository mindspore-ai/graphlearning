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
"""GatedGraphConv Layer"""
import math
import mindspore as ms
from mindspore._checkparam import Validator
from mindspore.common.initializer import XavierUniform
from mindspore_gl import Graph
from .. import GNNCell


class HomoGraphConv(GNNCell):
    """
    Homo Graph Conv

    Args:
        out_feat_size (int): Output node feature size.
        bias (bool): Whether use bias.

    Returns:
        Tensor, output node features.
    """

    def __init__(self,
                 out_feat_size: int,
                 bias=True):
        """
        Init HomoGraphConv.
        """
        super().__init__()
        gain = math.sqrt(2)  # gain for relu
        self.dense = ms.nn.Dense(out_feat_size, out_feat_size, has_bias=bias, weight_init=XavierUniform(gain))

    # pylint: disable=arguments-differ
    def construct(self, x, g: Graph):
        """
        Construct function for HomoGraphConv.
        """
        g.set_vertex_attr({"h": self.dense(x)})
        for v in g.dst_vertex:
            v.h = g.sum([u.h for u in v.innbs])
        return [v.h for v in g.dst_vertex]


class GatedGraphConv(ms.nn.Cell):
    r"""
    Gated Graph Convolution Layer. From the paper `Gated Graph Sequence Neural Networks
    <https://arxiv.org/pdf/1511.05493.pdf>`_ .

    .. math::
        h_{i}^{0} = [ x_i \| \mathbf{0} ] \\

        a_{i}^{t} = \sum_{j\in\mathcal{N}(i)} W_{e_{ij}} h_{j}^{t} \\

        h_{i}^{t+1} = \mathrm{GRU}(a_{i}^{t}, h_{i}^{t})

    Args:
        in_feat_size (int): Input node feature size.
        out_feat_size (int): Output node feature size.
        n_steps (int): Number of steps.
        n_etype (int): Number of edge types.
        bias (bool, optional): Whether use bias. Default: True.

    Inputs:
        - **x** (Tensor): The input node features. The shape is :math:`(N,*)` where :math:`N` is the number of nodes,
          and :math:`*` could be of any shape.
        - **src_idx** (List): The source index for each edge type.
        - **dst_idx** (List): The destination index for each edge type.
        - **n_nodes** (int): The number of nodes of the whole graph.
        - **n_edges** (List): The number of edges for each edge type.

    Outputs:
        - Tensor, output node features. The shape is :math:`(N, out\_feat\_size)`.

    Raises:
        TypeError: If `in_feat_size` is not a positive int.
        TypeError: If `out_feat_size` is not a positive int.
        TypeError: If `n_steps` is not a positive int.
        TypeError: If `n_etype` is not a positive int.
        TypeError: If `bias` is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore_gl.nn import GatedGraphConv
        >>> from mindspore_gl import GraphField
        >>> feat_size = 16
        >>> n_nodes = 4
        >>> h = ms.ops.Ones()((n_nodes, feat_size), ms.float32)
        >>> src_idx = [ms.Tensor([0, 1, 2, 3], ms.int32), ms.Tensor([0, 0, 1, 1], ms.int32),
        ...            ms.Tensor([0, 0, 1, 2, 3], ms.int32), ms.Tensor([2, 3, 3, 0, 1], ms.int32),
        ...            ms.Tensor([0, 1, 2, 3], ms.int32), ms.Tensor([2, 0, 2, 1], ms.int32)]
        >>> dst_idx = [ms.Tensor([0, 0, 1, 1], ms.int32), ms.Tensor([0, 1, 2, 3], ms.int32),
        ...            ms.Tensor([2, 3, 3, 0, 1], ms.int32), ms.Tensor([0, 0, 1, 2, 3], ms.int32),
        ...            ms.Tensor([2, 0, 2, 1], ms.int32), ms.Tensor([0, 1, 2, 3], ms.int32)]
        >>> n_edges = [4, 4, 5, 5, 4, 4]
        >>> conv = GatedGraphConv(feat_size, 16, 2, 6, True)
        >>> ret = conv(h, src_idx, dst_idx, n_nodes, n_edges)
        >>> print(ret.shape)
        (4, 16)
    """

    def __init__(self,
                 in_feat_size: int,
                 out_feat_size: int,
                 n_steps: int,
                 n_etype: int,
                 bias=True):

        super().__init__()
        if in_feat_size > out_feat_size:
            raise TypeError("GatedGraphConv requires input feature size <= out_feature_size")
        self.in_feat_size = in_feat_size
        self.out_feat_size = out_feat_size
        self.in_feat_size = Validator.check_positive_int(in_feat_size, "in_feat_size", self.cls_name)
        self.out_feat_size = Validator.check_positive_int(out_feat_size, "out_feat_size", self.cls_name)
        cl = []
        for _ in range(n_etype):
            cl.append(HomoGraphConv(out_feat_size, bias))
        self.cell_list = ms.nn.CellList(cl)
        self.n_etype = n_etype
        self.n_steps = n_steps
        self.gru = ms.nn.GRUCell(out_feat_size, out_feat_size)

    # pylint: disable=arguments-differ
    def construct(self, x, src_idx, dst_idx, n_nodes, n_edges):
        """
        Construct function for GatedGraphConv.
        """
        if self.in_feat_size < self.out_feat_size:
            x = ms.ops.Concat(axis=-1)(x, ms.ops.Zeros()((ms.ops.Shape()(x)[0], self.out_feat_size - self.in_feat_size),
                                                         ms.float32))
        for _ in range(self.n_steps):
            out = self.cell_list[0](x, src_idx[0], dst_idx[0], n_nodes, n_edges)
            for i in range(1, self.n_etype):
                out += self.cell_list[i](x, src_idx[i], dst_idx[i], n_nodes, n_edges)
            if self.gru is not None:
                out = self.gru(out, x)
                x = out
            x = out
        return x
