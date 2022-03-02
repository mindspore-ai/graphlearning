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
"""bgcf conv."""
import mindspore as ms
import mindspore.nn as nn
from mindspore import Parameter
from mindspore.common.initializer import initializer
from mindspore.common.initializer import XavierUniform
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

from mindspore_gl import Graph
from .. import GNNCell


class AttenConv(nn.Cell):
    """Attention conv"""

    def __init__(self,
                 in_feat_size: int,
                 out_size: int,
                 input_drop_out_rate: float = 1.0) -> None:
        super().__init__()
        self.in_feat_size = in_feat_size
        self.out_size = out_size
        self.fc = ms.nn.Dense(in_feat_size * 2, out_size, weight_init=XavierUniform(), has_bias=False)
        self.bias = ms.Parameter(initializer('zero', [out_size], ms.float32), name='bias')
        self.feat_drop = ms.nn.Dropout(1 - input_drop_out_rate)

        self.expanddims = P.ExpandDims()
        self.matmul = P.MatMul()
        self.matmul_3 = P.BatchMatMul()
        self.matmul_t = P.BatchMatMul(transpose_b=True)
        self.softmax = P.Softmax(axis=-1)
        self.concat = P.Concat(axis=1)
        self.gather = ms.ops.Gather()
        self.dropout = nn.Dropout(keep_prob=1 - input_drop_out_rate)
        self.out_weight = Parameter(
            initializer("XavierUniform", [in_feat_size * 2, out_size], dtype=mstype.float32))
        self.squeeze = P.Squeeze(1)

    # pylint: disable=arguments-differ
    def construct(self, node_feat, self_idx, neigth_idx):
        """attention conv forward"""
        self_feature = self.gather(node_feat, self_idx, 0)
        neigh_feature = self.gather(node_feat, neigth_idx, 0)
        query = self.expanddims(self_feature, 1)
        neigh_matrix = self.dropout(neigh_feature)

        score = self.matmul_t(query, neigh_matrix)
        score = self.softmax(score)
        atten_agg = self.matmul_3(score, neigh_matrix)
        atten_agg = self.squeeze(atten_agg)

        output = self.matmul(self.concat((atten_agg, self_feature)), self.out_weight)
        return output


class BGCFConv(GNNCell):
    """bgcf conv"""

    def __init__(self,
                 in_feat_size: int,
                 out_size: int,
                 input_drop_out_rate: float = 1.0) -> None:
        super().__init__()
        self.in_feat_size = in_feat_size
        self.out_size = out_size
        self.fc = ms.nn.Dense(in_feat_size * 2, out_size, weight_init=XavierUniform(), has_bias=False)
        self.bias = ms.Parameter(initializer('zero', [out_size], ms.float32), name='bias')
        self.feat_drop = ms.nn.Dropout(1 - input_drop_out_rate)

        self.expanddims = P.ExpandDims()
        self.exp = ms.ops.Exp()
        self.concat = P.Concat(axis=1)
        self.gather = ms.ops.Gather()
        self.squeeze = P.Squeeze(1)
        self.sum = P.ReduceSum(keep_dims=True)

    # pylint: disable=arguments-differ
    def construct(self, node_feat, self_idx, g: Graph):
        """bgcf conv forward"""
        feat_src = feat_dst = node_feat
        ed = feat_dst
        es = self.feat_drop(feat_src)
        g.set_vertex_attr({'es': es, 'ed': ed, 'feat_src': feat_src})
        for v in g.dst_vertex:
            relation = [self.sum(u.es * v.ed, -1) for u in v.innbs]
            edge = self.exp(self.squeeze(relation))
            attn = [c / g.sum(edge) for c in edge]
            feat = [u.es for u in v.innbs]
            v.h = g.sum(self.expanddims(attn, 1) * feat)
        output = self.fc(self.concat((self.gather([v.h for v in g.dst_vertex], self_idx, 0),
                                      self.gather([v.feat_src for v in g.dst_vertex], self_idx, 0))))
        return output
