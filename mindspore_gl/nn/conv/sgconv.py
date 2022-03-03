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

"""SGConv Layer"""
import math
import mindspore as ms
from mindspore import nn
from mindspore._checkparam import Validator
from mindspore.common.initializer import XavierUniform
from mindspore_gl import Graph
from .. import GNNCell


class SGConv(GNNCell):
    r"""
    Simplified Graph convolutional layer.
    From the paper `Simplifying Graph Convolutional Networks <https://arxiv.org/pdf/1902.07153.pdf>`_.

    .. math::
        H^{K} = (\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2})^K X \Theta

    Where :math:`$\tilde{A}=A+I`.

    Args:
        in_feat_size (int): Input node feature size.
        out_feat_size (int): Output node feature size.
        num_hops (int): Number of hops. Default: 1.
        cached (bool): Whether use cached. Default: True.
        bias (bool): Whether use bias. Default: True.
        norm (Cell): Normalization function Cell. Default: None.

    Inputs:
        - **x** (Tensor) - The input node features. The shape is :math:`(N,D\_in)`
          where :math:'N' is the number of nodes and :math:`D\_in` could be of any shape.
        - **in_deg** (Tensor) -  In degree for nodes. The shape is :math:'(N, )'
          where :math:'N' is the number of nodes.
        - **out_deg** (Tensor) -  Out degree for nodes. The shape is :math:'(N, )'
          where :math:'N' is the number of nodes.
        - **g** (Graph) - The input graph.

    Outputs:
        Tensor, the output feature of shape :math:'(N,D\_out)'
        where : math:'N' is the number of nodes and :math:`D\_out` could be of any shape.

    Raises:
        TypeError: if norm type is not ms.nn.Cell

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.context as context
        >>> from mindspore_gl.nn.conv import SGConv
        >>> from mindspore_gl import GraphField
        >>> context.set_context(device_target="GPU", mode=context.PYNATIVE_MODE)
        >>> n_nodes = 4
        >>> n_edges = 8
        >>> src_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 2, 3], ms.int32)
        >>> dst_idx = ms.Tensor([0, 1, 3, 1, 2, 3, 3, 2], ms.int32)
        >>> in_deg = ms.Tensor([1, 2, 2, 3], ms.int32)
        >>> out_deg = ms.Tensor([3, 3, 1, 1], ms.int32)
        >>> feat_size = 4
        >>> in_feat_size = feat_size
        >>> nh = ms.ops.Ones()((n_nodes, feat_size), ms.float32)
        >>> eh = ms.ops.Ones()((n_edges, feat_size), ms.float32)
        >>> g = GraphField(src_idx, dst_idx, 4, 8)
        >>> in_deg = in_deg
        >>> out_deg = out_deg
        >>> sgconv = SGConv(in_feat_size, 4)
        >>> res = sgconv(nh, in_deg, out_deg, *g.get_graph())
        >>> print(res.shape)
            (4, 4)
    """

    def __init__(self,
                 in_feat_size: int,
                 out_feat_size: int,
                 num_hops: int = 1,
                 cached: bool = True,
                 bias: bool = True,
                 norm: ms.nn.Cell = None):
        super().__init__()
        in_feat_size = Validator.check_positive_int(in_feat_size, "in_feat_size", self.cls_name)
        out_feat_size = Validator.check_positive_int(out_feat_size, "out_feat_size", self.cls_name)
        self.num_hops = Validator.check_positive_int(num_hops, "num_hops", self.cls_name)
        self.cached = Validator.check_bool(cached, "cached", self.cls_name)
        bias = Validator.check_bool(bias, "bias", self.cls_name)

        if norm:
            if not isinstance(norm, nn.Cell):
                raise TypeError("norm type should be ms.nn.Cell")

        self.dense = ms.nn.Dense(in_feat_size, out_feat_size, has_bias=bias, weight_init=XavierUniform(math.sqrt(2)))
        self.cached_h = None
        self.norm = norm
        self.min_clip = ms.Tensor(1, ms.int32)
        self.max_clip = ms.Tensor(100000000, ms.int32)

    # pylint: disable=arguments-differ
    def construct(self, x, in_deg, out_deg, g: Graph):
        """
        Construct function for SGConv.
        """
        feat = x
        if self.cached_h:
            feat = self.cached_h
        else:
            in_deg = ms.ops.clip_by_value(in_deg, self.min_clip, self.max_clip)
            in_deg = ms.ops.Reshape()(ms.ops.Pow()(in_deg, -0.5), ms.ops.Shape()(in_deg) + (1,))
            out_deg = ms.ops.clip_by_value(out_deg, self.min_clip, self.max_clip)
            out_deg = ms.ops.Reshape()(ms.ops.Pow()(out_deg, -0.5), ms.ops.Shape()(out_deg) + (1,))
            for _ in range(self.num_hops):
                feat = feat * out_deg
                g.set_vertex_attr({"h": feat})
                for v in g.dst_vertex:
                    v.h = g.sum([u.h for u in v.innbs])
                feat = [v.h for v in g.dst_vertex] * in_deg
            if self.norm is not None:
                feat = self.norm(feat)
            if self.cached:
                self.cached_h = feat
        return self.dense(feat)
