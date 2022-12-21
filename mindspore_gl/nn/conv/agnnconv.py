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
"""AGNNConv Layer."""
import mindspore as ms
from mindspore._checkparam import Validator
from mindspore_gl import Graph
from .. import GNNCell


class AGNNConv(GNNCell):
    r"""
    Attention Based Graph Neural Network.
    From the paper `Attention-based Graph Neural Network for Semi-Supervised Learning <https://arxiv.org/abs/1803.03735>`_ .

    .. math::
        H^{l+1} = P H^{l}

    Computation of :math:`P` is:

    .. math::
        P_{ij} = \mathrm{softmax}_i ( \beta \cdot \cos(h_i^l, h_j^l))

    :math:`\beta` is a single scalar parameter.

    Args:
        init_beta (float): Init :math:`\beta`, a single scalar parameter. Default: 1.0.
        learn_beta (bool): Whether :math:`\beta` is learnable. Default: True.

    Inputs:
        - **x** (Tensor): The input node features. The shape is :math:`(N,*)` where :math:`N` is the number of nodes,
          and :math:`*` could be of any shape.
        - **g** (Graph): The input graph.

    Outputs:
        - Tensor, output node features, where the shape should be the same as input 'x'.

    Raises:
        TypeError: If 'init_beta' is not a float.
        TypeError: If 'learn_beta' is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore_gl.nn import AGNNConv
        >>> from mindspore_gl import GraphField
        >>> n_nodes = 4
        >>> n_edges = 8
        >>> feat_size = 16
        >>> src_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 2, 3], ms.int32)
        >>> dst_idx = ms.Tensor([0, 1, 3, 1, 2, 3, 3, 2], ms.int32)
        >>> ones = ms.ops.Ones()
        >>> feat = ones((n_nodes, feat_size), ms.float32)
        >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
        >>> conv = AGNNConv()
        >>> ret = conv(feat, *graph_field.get_graph())
        >>> print(ret.shape)
        (4, 16)
    """

    def __init__(self,
                 init_beta: float = 1.,
                 learn_beta: bool = True):
        super().__init__()
        init_beta = Validator.check_is_float(init_beta, "init_beta", self.cls_name)
        learn_beta = Validator.check_bool(learn_beta, 'learn_beta', self.cls_name)
        if learn_beta:
            self.beta = ms.Parameter(ms.Tensor([init_beta], ms.float32))
        else:
            self.beta = ms.Tensor([init_beta], ms.float32)

    # pylint: disable=arguments-differ
    def construct(self, x, g: Graph):
        """
        Construct function for AGNNConv.
        """
        g.set_vertex_attr({"h": x, "norm_h": ms.ops.L2Normalize()(x)})
        for v in g.dst_vertex:
            cosine_dis = [ms.ops.Exp()(self.beta * g.dot(u.norm_h, v.norm_h)) for u in v.innbs]
            a = cosine_dis / g.sum(cosine_dis)
            v.h = g.sum([u.h for u in v.innbs] * a)
        return [v.h for v in g.dst_vertex]
