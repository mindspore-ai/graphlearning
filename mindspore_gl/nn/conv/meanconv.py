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
"""MeanConv Layer"""
import mindspore as ms
from mindspore.ops import operations as P
from mindspore.common.initializer import XavierUniform
from mindspore_gl import Graph
from .. import GNNCell


class MeanConv(GNNCell):
    r"""
    GraphSAGE Layer, from the paper `Inductive Representation Learning on Large Graphs
    <https://arxiv.org/pdf/1706.02216.pdf>`_.

    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} = \mathrm{aggregate}
        \left(\{h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right) \\

        h_{i}^{(l+1)} = \sigma \left(W \cdot \mathrm{concat}
        (h_{i}^{l}, h_{\mathcal{N}(i)}^{l+1}) \right)\\

        h_{i}^{(l+1)} = \mathrm{norm}(h_{i}^{l})

    If weights are provided on each edge, the weighted graph convolution is defined as:

    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} = \mathrm{aggregate}
        \left(\{e_{ji} h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)

    Args:
        in_feat_size (int): Input node feature size.
        out_feat_size (int): Output node feature size.
        aggregator_type (str): Type of aggregator, should in 'pool', 'lstm' and 'mean'.
        feat_drop (float): Feature drop out rate.
        bias (bool): Whether use bias.
        norm (Cell): Normalization function Cell, default is None.
        activation (Cell): Activation function Cell, default is None.

    Raises:
        SyntaxError: when aggregator type not supported.
    """

    def __init__(self,
                 in_feat_size: int,
                 out_feat_size: int,
                 feat_drop=0.6,
                 bias=False,
                 norm=None,
                 activation: ms.nn.Cell = None):
        super().__init__()
        self.in_feat_size = in_feat_size
        self.out_feat_size = out_feat_size
        self.norm = norm
        if activation == "tanh":
            self.activation = P.Tanh()
        elif activation == "relu":
            self.activation = P.ReLU()
        else:
            raise ValueError("activation should be tanh or relu")
        self.feat_drop = ms.nn.Dropout(1 - feat_drop)
        self.concat = P.Concat(axis=1)
        if bias:
            self.bias = ms.Parameter(ms.ops.Zeros()(self.out_feat_size, ms.float32))
        else:
            self.bias = None
        self.dense_self = ms.nn.Dense(self.in_feat_size * 2, self.out_feat_size, has_bias=False,
                                      weight_init=XavierUniform())
        self.gather = ms.ops.Gather()

    # pylint: disable=arguments-differ
    def construct(self, node_feat, self_idx, g: Graph):
        """
        Construct function for SAGEConv.

        Args:
            node_feat (Tensor): The input node features.
            self_idx(int): The node idx
            g (Graph): The input graph.

        Returns:
            Tensor, output node features.
        """
        # node_feat = self.feat_drop(node_feat)
        g.set_vertex_attr({"src": node_feat})
        for v in g.dst_vertex:
            v.rst = self.feat_drop(g.avg([u.src for u in v.innbs]))
        ret = self.dense_self(self.concat((self.gather([v.src for v in g.dst_vertex], self_idx, 0),
                                           self.gather([v.rst for v in g.dst_vertex], self_idx, 0))))
        if self.bias is not None:
            ret = ret + self.bias
        if self.activation is not None:
            ret = self.activation(ret)
        if self.norm is not None:
            ret = self.norm(self.ret)
        return ret
