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
"""gnn module"""
import mindspore as ms
import mindspore.ops as ops
from mindspore_gl import BatchedGraph
from mindspore_gl.nn import GNNCell
from conv import GCNConv, GINConv


class GNNVirtualNodeMS(GNNCell):
    """
        Virtual GNN to generate node embedding
    """

    def __init__(self, num_layer, emb_dim, node_encoder,
                 edge_encoder_cls, drop_ratio=0.5, jk="last", residual=False,
                 gnn_type="gcn"):
        super(GNNVirtualNodeMS, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.jk = jk
        self.residual = residual
        self.gnn_type = gnn_type
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = node_encoder
        if edge_encoder_cls is not None:
            self.edge_encoder = edge_encoder_cls(emb_dim)
        else:
            self.edge_encoder = None
        self.virtualnode_embedding = ms.nn.Embedding(1, emb_dim)
        constant_init = ms.common.initializer.Constant(value=0)
        constant_init(self.virtualnode_embedding.embedding_table[0])
        self.conv_list = ms.nn.CellList()
        self.batch_normalize_list = ms.nn.CellList()
        self.mlp_virtualnode_list = ms.nn.CellList()
        self.dropout_list = ms.nn.CellList()
        self.dropout_virtual_list = ms.nn.CellList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.conv_list.append(
                    GINConv(ms.nn.ReLU()))
            elif gnn_type == "gcn":
                self.conv_list.append(
                    GCNConv(emb_dim, emb_dim,
                            activation=ms.nn.ReLU(), dropout=self.drop_ratio))
            else:
                ValueError("Undefined GNN type called {}".format(gnn_type))
            self.batch_normalize_list.append(
                ms.nn.SequentialCell(
                    [ms.nn.BatchNorm1d(node_feat_size=emb_dim, momentum=0.9)]))

        for layer in range(num_layer):
            if layer == num_layer - 1:
                self.dropout_list.append(
                    ms.nn.SequentialCell([ms.nn.Dropout(p=1 - self.drop_ratio)]))
            else:
                self.dropout_list.append(
                    ms.nn.SequentialCell(
                        [ms.nn.ReLU(), ms.nn.Dropout(p=1 - self.drop_ratio)]))
                self.dropout_virtual_list.append(
                    ms.nn.SequentialCell([ms.nn.Dropout(p=1 - self.drop_ratio)]))
                self.mlp_virtualnode_list.append(
                    ms.nn.SequentialCell(
                        [ms.nn.Dense(emb_dim, 2 * emb_dim),
                         ms.nn.BatchNorm1d(2 * emb_dim, momentum=0.1),
                         ms.nn.ReLU(),
                         ms.nn.Dense(2 * emb_dim, emb_dim),
                         ms.nn.BatchNorm1d(emb_dim, momentum=0.1),
                         ms.nn.ReLU()]))

    def construct(self, batched_data, perturb, bg: BatchedGraph):
        """gnn module forward"""
        x, edge_index, edge_attr, batch, edge_weight, _ = batched_data
        in_degree, out_degree = \
            self.get_degree_func(bg.in_degree(), bg.out_degree())

        node_depth = batched_data.node_depth \
            if hasattr(batched_data, "node_depth") else None
        if self.node_encoder is not None:
            encoded_node = (
                self.node_encoder(
                    x) if node_depth is None else self.node_encoder(
                        x, node_depth.view(-1,),))
        else:
            encoded_node = x
        tmp = encoded_node + perturb \
            if perturb is not None else encoded_node
        h_list = [tmp]

        if self.edge_encoder is not None:
            edge_weight = self.edge_encoder(edge_attr)
        zeros = ops.Zeros()
        temp_zeros = zeros(
            (batch.asnumpy().tolist()[-1] + 1,), edge_index.dtype)
        virtualnode_embedding = self.virtualnode_embedding(temp_zeros)
        for layer in range(self.num_layer):
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]
            if self.gnn_type == "gin":
                h = self.conv_list[layer](h_list[layer], edge_weight, bg)
            elif self.gnn_type == "gcn":
                h = self.conv_list[layer](h_list[layer],
                                          in_degree, out_degree, bg)
            else:
                ValueError("Undefined GNN type called {}".format(self.gnn_type))
            hidden = self.batch_normalize_list[layer](h)
            if layer == self.num_layer - 1:
                hidden = self.dropout_list[layer](hidden)
            else:
                hidden = self.dropout_list[layer](hidden)

            if self.residual:
                hidden = hidden + h_list[layer]
            h_list.append(hidden)
            if layer < self.num_layer - 1:
                hypotheticnode_embedding_middle = \
                    bg.sum_nodes(x) + virtualnode_embedding
                if self.residual:
                    hypotheticnode_embedding_middle = \
                        self.mlp_virtualnode_list[layer](
                            hypotheticnode_embedding_middle)
                    virtualnode_embedding = \
                        virtualnode_embedding + \
                        self.dropout_virtual_list[layer](
                            hypotheticnode_embedding_middle)
                else:
                    hypotheticnode_embedding_middle = \
                        self.mlp_virtualnode_list[layer](
                            hypotheticnode_embedding_middle)
                    virtualnode_embedding = \
                        self.dropout_virtual_list[layer](
                            hypotheticnode_embedding_middle)

        return self.get_node_embedding(h_list)

    def get_node_embedding(self, h_list):
        """
        After layers, get graph node embedding.

        Args:
            h_list: layer list

        Returns: node_representation

        """
        if self.jk == "last":
            node_representation = h_list[-1]
        elif self.jk == "sum":
            node_representation = 0
            for layer in range(self.num_layer):
                node_representation += h_list[layer]
        elif self.jk == "cat":
            concat_op = ms.ops.Concat(-1)
            node_representation = concat_op([h_list[0], h_list[-1]])

        return node_representation

    @staticmethod
    def get_degree_func(in_degree, out_degree):
        out_degree = ops.transpose(out_degree, (1, 0))
        in_degree = ops.transpose(in_degree, (1, 0))
        squeeze = ops.Squeeze()
        out_degree = squeeze(out_degree)
        in_degree = squeeze(in_degree)

        return in_degree, out_degree


def gnn_node_embedding(virtual_node, *args, **kwargs):
    if not virtual_node:
        raise NotImplementedError

    return GNNVirtualNodeMS(*args, **kwargs)
