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
from mindspore._checkparam import Validator
from mindspore_gl import Graph
from .. import GNNCell


class GINConv(GNNCell):
    r"""
    Graph isomorphic network layer.
    From the paper `How Powerful are Graph Neural Networks? <https://arxiv.org/pdf/1810.00826.pdf>`_ .

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
        activation (mindspore.nn.Cell): Activation function.
        init_eps (float, optional): Init value of eps. Default: 0.
        learn_eps (bool, optional): Whether eps is learnable. Default: False.
        aggregation_type (str, optional): Type of aggregation, should in `'sum'`, `'max'` and `'avg'`.
            Default: 'sum'.

    Inputs:
        - **x** (Tensor): The input node features. The shape is :math:`(N,*)` where :math:`N` is the number of nodes,
          and :math:`*` could be of any shape.
        - **edge_weight** (Tensor): The input edge weights. The shape is :math:`(M,*)` where :math:`M` is the number
          of nodes, and :math:`*` could be of any shape.
        - **g** (Graph): The input graph.
    Outputs:
        - Tensor, output node features. The shape is :math:`(N, out\_feat\_size)`.

    Raises:
        TypeError: If `activation` is not a mindspore.nn.Cell.
        TypeError: If `init_eps` is not a float.
        TypeError: If `learn_eps` is not a bool.
        SyntaxError: Raised when the `aggregation_type` not in `'sum'`, `'max'` and `'avg'`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore_gl.nn import GINConv
        >>> from mindspore_gl import GraphField
        >>> n_nodes = 4
        >>> n_edges = 8
        >>> feat_size = 16
        >>> src_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 2, 3], ms.int32)
        >>> dst_idx = ms.Tensor([0, 1, 3, 1, 2, 3, 3, 2], ms.int32)
        >>> ones = ms.ops.Ones()
        >>> nodes_feat = ones((n_nodes, feat_size), ms.float32)
        >>> edges_weight = ones((n_edges, feat_size), ms.float32)
        >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
        >>> conv = GINConv(activation=None, init_eps=0., learn_eps=False, aggregation_type="sum")
        >>> ret = conv(nodes_feat, edges_weight, *graph_field.get_graph())
        >>> print(ret.shape)
        (4, 16)
    """

    def __init__(self,
                 activation,
                 init_eps=0.,
                 learn_eps=False,
                 aggregation_type="sum"):
        super().__init__()
        init_eps = Validator.check_is_float(init_eps, "init_eps", self.cls_name)
        learn_eps = Validator.check_bool(learn_eps, "learn_eps", self.cls_name)
        if activation is not None and not isinstance(activation, ms.nn.Cell):
            raise TypeError(f"For '{self.cls_name}', the 'activation' must a mindspore.nn.Cell, but got "
                            f"{type(activation).__name__}.")
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
