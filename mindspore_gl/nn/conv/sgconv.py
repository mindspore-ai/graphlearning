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
"""SGConv Layer."""
import math
import mindspore as ms
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
        num_hops (int): Number of hops.
        cached (bool): Whether use cached.
        bias (bool): Whether use bias.
        norm (Cell): Normalization function Cell, default is None.
    """

    def __init__(self,
                 in_feat_size: int,
                 out_feat_size: int,
                 num_hops: int = 1,
                 cached: bool = True,
                 bias: bool = True,
                 norm: ms.nn.Cell = None):
        super().__init__()
        self.dense = ms.nn.Dense(in_feat_size, out_feat_size, has_bias=bias, weight_init=XavierUniform(math.sqrt(2)))
        self.cached = cached
        self.cached_h = None
        self.num_hops = num_hops
        self.norm = norm
        self.min_clip = ms.Tensor(1, ms.int32)
        self.max_clip = ms.Tensor(100000000, ms.int32)

    # pylint: disable=arguments-differ
    def construct(self, x, in_deg, out_deg, g: Graph):
        """
        Construct function for SGConv.

        Args:
            x (Tensor): The input node features.
            in_deg (Tensor): In degree for nodes.
            out_deg (Tensor): Out degree for nodes.
            g (Graph): The input graph.

        Returns:
            Tensor, output node features.
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
