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
from mindspore.common.initializer import XavierUniform
from mindspore_gl import Graph
from .. import GNNCell


class HomoGraphConv(GNNCell):
    """
    Homo Graph Conv

    Args:
            out_feat_size (int): Output node feature size.
            bias (bool): Whether use bias.
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

        Args:
            x (Tensor): The input node features.
            g (Graph): The input graph.

        Returns:
            Tensor, output node features.
        """
        g.set_vertex_attr({"h": self.dense(x)})
        for v in g.dst_vertex:
            v.h = g.sum([u.h for u in v.innbs])
        return [v.h for v in g.dst_vertex]


class GatedGraphConv(ms.nn.Cell):
    r"""
    Gated Graph Convolution Layer, from the paper `Gated Graph Sequence Neural Networks
    <https://arxiv.org/pdf/1511.05493.pdf>`_.

    .. math::
        h_{i}^{0} = [ x_i \| \mathbf{0} ] \\

        a_{i}^{t} = \sum_{j\in\mathcal{N}(i)} W_{e_{ij}} h_{j}^{t} \\

        h_{i}^{t+1} = \mathrm{GRU}(a_{i}^{t}, h_{i}^{t})

    Args:
        in_feat_size (int): Input node feature size.
        out_feat_size (int): Output node feature size.
        n_steps (int): Number of steps.
        n_etype (int): Number of edge types.
        bias (bool): Whether use bias.
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
        cl = []
        for _ in range(n_etype):
            cl.append(HomoGraphConv(out_feat_size, bias))
        self.cell_list = ms.nn.CellList(cl)
        self.n_etype = n_etype
        self.n_steps = n_steps
        self.gru = ms.nn.GRU(out_feat_size, out_feat_size)

    # pylint: disable=arguments-differ
    def construct(self, x, src_idx, dst_idx, n_nodes, n_edges):
        r"""
        Construct function for GatedGraphConv.

        Args:
            x (Tensor): The input node features.
            src_idx (Tensor): A tensor with shape :math:`(N\_EDGES)`,
                        represents the source node index of COO edge matrix.
            dst_idx (Tensor): A tensor with shape :math:`(N\_EDGES)`,
                        represents the destination node index of COO edge matrix.
            n_nodes (Tensor): An integer tensor.
            n_edges (Tensor): An integer tensor.

        Returns:
            Tensor, output node features.
        """
        if self.in_feat_size < self.out_feat_size:
            x = ms.ops.Concat(axis=-1)(x, ms.ops.Zeros()((ms.ops.Shape()(x)[0], self.out_feat_size - self.in_feat_size),
                                                         ms.float32))
        for _ in range(self.n_steps):
            out = self.cell_list[0](x, src_idx[0], dst_idx[0], n_nodes, n_edges)
            for i in range(1, self.n_etype):
                out += self.cell_list[i](x, src_idx[i], dst_idx[i], n_nodes, n_edges)
            if self.gru is not None:
                out = self.gru(out)
            x = out
        return out
