# Copyright 2021 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""graphsage Net"""
import math
import mindspore as ms
from mindspore.nn import Cell
from mindspore.common.initializer import XavierUniform
from mindspore_gl.nn import SAGEConv

class SAGENet(Cell):
    """graphsage net"""
    def __init__(self, in_feat_size, hidden_feat_size, appr_feat_size, out_feat_size):
        super().__init__()
        self.num_layers = 2
        self.layer1 = SAGEConv(in_feat_size, hidden_feat_size, aggregator_type='mean')
        self.layer2 = SAGEConv(hidden_feat_size, appr_feat_size, aggregator_type='mean')
        self.dense_out = ms.nn.Dense(appr_feat_size, out_feat_size, has_bias=False,
                                     weight_init=XavierUniform(math.sqrt(2)))
        self.activation = ms.nn.ReLU()
        self.dropout = ms.nn.Dropout(0.5)

    def construct(self, node_feat, edges, n_nodes, n_edges):
        """graphsage net forward"""
        node_feat = self.layer1(node_feat, None, edges[0], edges[1], n_nodes, n_edges)
        node_feat = self.activation(node_feat)
        node_feat = self.dropout(node_feat)
        ret = self.layer2(node_feat, None, edges[0], edges[1], n_nodes, n_edges)
        ret = self.dense_out(ret)
        return ret
