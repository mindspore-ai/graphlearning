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
"""GATConv Layer"""
import math
import mindspore as ms
from mindspore._checkparam import Validator
from mindspore.common.initializer import initializer
from mindspore.common.initializer import XavierUniform
from mindspore_gl import Graph
from .. import GNNCell


class GATConv(GNNCell):
    r"""
    Graph Attention Network, from the paper `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`_.

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}

    :math:`\alpha_{i, j}` represents the attention score between node :math:`i` and node :math:`j`.

    .. math::
        \alpha_{ij}^{l} = \mathrm{softmax_i} (e_{ij}^{l}) \\
        e_{ij}^{l} = \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

    Args:
        in_feat_size (int): Input node feature size.
        out_size (int): Output node feature size.
        num_attn_head (int): Number of attention head used in GAT.
        input_drop_out_rate (float): Keep rate of input drop out. Default: 1.0.
        attn_drop_out_rate (float): Keep rate of attention drop out. Default: 1.0.
        leaky_relu_slope (float): Slope for leaky relu. Default: 0.2.
        activation (Cell): Activation function, default is None.
        add_norm: Whether the edge information needs normalization or not. Default: False.
    Inputs:
        - **x** (Tensor) - The input node features. The shape is :math:`(N,D_{in})`
          where :math:`N` is the number of nodes and :math:`D_{in}` could be of any shape.
        - **g** (Graph) - The input graph.

    Outputs:
        Tensor, the output feature of shape :math:`(N,D_{out})` where :math:`D_{out}` should be equal to
        :math:`D_{in} * num\_attn\_head`.

    Raises:
        TypeError: If `in_feat_size`, `out_size`, or `num_attn_head` is not an int.
        TypeError: If `input_drop_out_rate`, `attn_drop_out_rate`, or `leaky_relu_slope` is not a float.
        TypeError: If `activation` is not a Cell.
        ValueError: If `input_drop_out_rate` or `attn_drop_out_rate` is not in range (0.0, 1.0]

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore_gl.nn import GATConv
        >>> from mindspore_gl import GraphField
        >>> n_nodes = 4
        >>> n_edges = 7
        >>> feat_size = 4
        >>> src_idx = ms.Tensor([0, 1, 1, 2, 2, 3, 3], ms.int32)
        >>> dst_idx = ms.Tensor([0, 0, 2, 1, 3, 0, 1], ms.int32)
        >>> ones = ms.ops.Ones()
        >>> feat = ones((n_nodes, feat_size), ms.float32)
        >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
        >>> gatconv = GATConv(in_feat_size=4, out_size=2, num_attn_head=3)
        >>> res = gatconv(feat, *graph_field.get_graph())
        >>> print(res.shape)
        (4, 6)
    """

    def __init__(self,
                 in_feat_size: int,
                 out_size: int,
                 num_attn_head: int,
                 input_drop_out_rate: float = 1.0,
                 attn_drop_out_rate: float = 1.0,
                 leaky_relu_slope: float = 0.2,
                 activation=None,
                 add_norm=False) -> None:
        super().__init__()
        self.in_feat_size = Validator.check_positive_int(in_feat_size, "in_feat_size", self.cls_name)
        self.out_size = Validator.check_positive_int(out_size, "out_size", self.cls_name)
        self.num_attn_head = Validator.check_positive_int(num_attn_head, "num_attn_head", self.cls_name)
        input_drop_out_rate = Validator.check_is_float(input_drop_out_rate, "input_drop_out_rate", self.cls_name)
        attn_drop_out_rate = Validator.check_is_float(attn_drop_out_rate, "attn_drop_out_rate", self.cls_name)
        leaky_relu_slope = Validator.check_is_float(leaky_relu_slope, "leaky_relu_slope", self.cls_name)
        add_norm = Validator.check_bool(add_norm, "add_norm", self.cls_name)
        if input_drop_out_rate <= 0.0 or input_drop_out_rate > 1.0:
            raise ValueError(f"For '{self.cls_name}', the 'input_drop_out_rate' should be a number in range (0.0, 1.0],"
                             f"but got {input_drop_out_rate}.")
        if attn_drop_out_rate <= 0.0 or attn_drop_out_rate > 1.0:
            raise ValueError(f"For '{self.cls_name}', the 'attn_drop_out_rate' should be a number in range (0.0, 1.0],"
                             f"but got {attn_drop_out_rate}.")
        self.reshape = ms.ops.Reshape()
        gain = math.sqrt(2)  # gain for relu
        self.fc = ms.nn.Dense(in_feat_size, out_size * num_attn_head, weight_init=XavierUniform(gain), has_bias=False)
        self.attn_s = ms.Parameter(initializer(XavierUniform(gain), [num_attn_head, out_size], ms.float32),
                                   name="attn_s")
        self.attn_d = ms.Parameter(initializer(XavierUniform(gain), [num_attn_head, out_size], ms.float32),
                                   name="attn_d")
        self.bias = ms.Parameter(initializer('zero', [num_attn_head, out_size], ms.float32), name='bias')
        self.feat_drop = ms.nn.Dropout(input_drop_out_rate)
        self.attn_drop = ms.nn.Dropout(attn_drop_out_rate)
        self.leaky_relu = ms.nn.LeakyReLU(leaky_relu_slope)
        self.exp = ms.ops.Exp()
        if add_norm:
            self.norm_constant = ms.Tensor(100, ms.float32)
            self.norm_div = ms.ops.Div()
        else:
            self.norm_div = None
        self.activation = activation

    # pylint: disable=arguments-differ
    def construct(self, x, g: Graph):
        """
        Construct function for GATConv.
        """
        x = self.feat_drop(x)
        x = self.fc(x)
        feat_src = feat_dst = ms.ops.Reshape()(x, (-1, self.num_attn_head, self.out_size))
        ed = ms.ops.ReduceSum(True)(feat_dst * self.attn_d, -1)
        g.set_vertex_attr({'es': ms.ops.ReduceSum(True)(feat_src * self.attn_s, -1), 'ed': ed, 'feat_src': feat_src})
        for v in g.dst_vertex:
            if self.norm_div is not None:
                edge = [self.leaky_relu(u.es + v.ed) for u in v.innbs]
                edge = self.exp(self.norm_div(edge, self.norm_constant))
            else:
                edge = [self.exp(self.leaky_relu(u.es + v.ed)) for u in v.innbs]
            attn = self.attn_drop([c / g.sum(edge) for c in edge])
            feat = [u.feat_src for u in v.innbs]
            v.h = g.sum(attn * feat)
            v.h = v.h + self.bias
            if self.activation:
                v.h = self.activation(v.h)
        return ms.ops.Flatten()([v.h for v in g.dst_vertex])
