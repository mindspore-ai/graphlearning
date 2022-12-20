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
from mindspore._checkparam import Validator
from mindspore_gl import Graph
from .. import GNNCell


class MeanConv(GNNCell):
    r"""
    GraphSAGE Layer. From the paper `Inductive Representation Learning on Large Graphs
    <https://arxiv.org/pdf/1706.02216.pdf>`_ .

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
        feat_drop (float): The keep rate, greater than 0 and less equal than 1. E.g. dropout=0.9,
            dropping out 10% of input units. Default: 0.6.
        bias (bool): Whether use bias. Default: False.
        norm (Cell): Normalization function Cell. Default: None.
        activation (Cell): Activation function Cell. Default: None.

    Inputs:
        - **x** (Tensor) - The input node features. The shape is :math:`(N,D\_in)`
          where :math:`N` is the number of nodes and :math:`D\_in` could be of any shape.
        - **self_idx** (Tensor) - The node idx. The shape is :math:`(N\_v,)`
          where :math:`N\_v` is the number of self nodes.
        - **g** (Graph) - The input graph.

    Outputs:
        - Tensor, the output feature of shape :math:`(N\_v,D\_out)`.
          where :math:`N\_v` is the number of self nodes and :math:`D\_out` could be of any shape

    Raises:
        TypeError: If `in_feat_size` or `out_feat_size` is not an int.
        TypeError: If `bias` is not a bool.
        TypeError: If `norm` is not a mindspore.nn.Cell.
        ValueError: If `dropout` is not in range (0.0, 1.0]
        ValueError: If `activation` is not tanh or relu.

    Supported Platforms:
        ``GPU`` ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore_gl.nn import MeanConv
        >>> from mindspore_gl import GraphField
        >>> n_nodes = 4
        >>> n_edges = 7
        >>> feat_size = 4
        >>> src_idx = ms.Tensor([0, 1, 1, 2, 2, 3, 3], ms.int32)
        >>> dst_idx = ms.Tensor([0, 0, 2, 1, 3, 0, 1], ms.int32)
        >>> ones = ms.ops.Ones()
        >>> feat = ones((n_nodes, feat_size), ms.float32)
        >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
        >>> gmmconv = MeanConv(in_feat_size=4, out_feat_size=2, activation='relu')
        >>> self_idx = ms.Tensor([0, 1], ms.int32)
        >>> res = gmmconv(feat, self_idx, *graph_field.get_graph())
        >>> print(res.shape)
        (2, 2)
    """

    def __init__(self,
                 in_feat_size: int,
                 out_feat_size: int,
                 feat_drop=0.6,
                 bias=False,
                 norm=None,
                 activation: ms.nn.Cell = None):
        super().__init__()
        self.in_feat_size = Validator.check_positive_int(in_feat_size, "in_feat_size", self.cls_name)
        self.out_feat_size = Validator.check_positive_int(out_feat_size, "in_feat_size", self.cls_name)
        bias = Validator.check_bool(bias, "bias", self.cls_name)
        self.norm = norm
        if activation == "tanh":
            self.activation = P.Tanh()
        elif activation == "relu":
            self.activation = P.ReLU()
        else:
            raise ValueError("activation should be tanh or relu")
        if dropout <= 0.0 or dropout > 1.0:
            raise ValueError(f"For '{self.cls_name}', the 'keep_prob' should be a number in range (0.0, 1.0], "
                             f"but got {dropout}.")
        if norm is not None and not isinstance(norm, Cell):
            raise TypeError(f"For '{self.cls_name}', the 'activation' must a mindspore.nn.Cell, but got "
                            f"{type(norm).__name__}.")
        self.feat_drop = ms.nn.Dropout(feat_drop)
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
        Construct function for MEANConv.
        """
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
