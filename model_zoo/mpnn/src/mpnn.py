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
"""MPNN networks"""
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore_gl.nn import GNNCell
from mindspore_gl import BatchedGraph
from mindspore_gl.nn.conv import NNConv
from mindspore_gl.nn.glob import Set2Set


# this network provide the message passing and vertex update function.
class MPNNGNN(GNNCell):
    """MPNN GNN"""
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats=64,
                 edge_hidden_feats=128,
                 num_step_message_passing=6):
        super().__init__()
        self.project_node_feats = nn.SequentialCell(
            nn.Dense(node_in_feats, node_out_feats),
            nn.ReLU()
        )
        self.num_step_message_passing = num_step_message_passing
        edge_network = nn.SequentialCell(
            nn.Dense(edge_in_feats, edge_hidden_feats),
            nn.ReLU(),
            nn.Dense(edge_hidden_feats, node_out_feats * node_out_feats)
        )
        self.gnn_layer = NNConv(
            in_feat_size=node_out_feats,
            out_feat_size=node_out_feats,
            edge_embed=edge_network,
            aggregator_type='sum'
        )
        self.gru = ms.nn.GRU(node_out_feats, node_out_feats)

    def construct(self, node_feats, edge_feats, bg: BatchedGraph):
        node_feats = self.project_node_feats(node_feats)
        hidden_feats = ops.ExpandDims()(node_feats, 0)

        for _ in range(self.num_step_message_passing):
            node_feats = nn.ReLU()(self.gnn_layer(node_feats, edge_feats, bg))
            node_feats, hidden_feats = self.gru(ops.ExpandDims()(node_feats, 0), hidden_feats)
            node_feats = ops.Squeeze(0)(node_feats)
        return node_feats


class MPNNPredictor(GNNCell):
    """MPNN Predictor"""
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats=64,
                 edge_hidden_feats=128,
                 n_tasks=1,
                 num_step_message_passing=6,
                 num_step_set2set=6,
                 num_layer_set2set=3):
        super().__init__()
        self.gnn = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)
        self.readout = Set2Set(input_size=node_out_feats,
                               num_iters=num_step_set2set,
                               num_layers=num_layer_set2set)
        self.predict = nn.SequentialCell(
            nn.Dense(2 * node_out_feats, node_out_feats),
            nn.ReLU(),
            nn.Dense(node_out_feats, n_tasks)
        )

    def construct(self, node_feats, edge_feats, bg: BatchedGraph):
        node_feats = self.gnn(node_feats, edge_feats, bg)
        graph_feats = self.readout(node_feats, bg)
        return self.predict(graph_feats)
