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
"""Dataset"""
import numpy as np
import mindspore as ms
from mindspore_gl import BatchedGraphField
from mindspore_gl.graph import BatchHomoGraph, PadArray2d, PadHomoGraph, PadMode, PadDirection
from mindspore_gl.dataloader import Dataset


class MultiHomoGraphDataset(Dataset):
    """MultiHomo Graph Dataset"""
    def __init__(self, dataset, batch_size, mode=PadMode.CONST, node_size=40, edge_size=1000, length=None):
        self._dataset = dataset
        self._batch_size = batch_size
        self.batch_fn = BatchHomoGraph()
        self.batched_edge_feat = None
        self.length = length
        if mode == PadMode.CONST:
            self.node_feat_pad_op = PadArray2d(dtype=np.float32, mode=PadMode.CONST, direction=PadDirection.COL,
                                               size=(node_size, dataset.num_features), fill_value=0)
            self.edge_feat_pad_op = PadArray2d(dtype=np.float32, mode=PadMode.CONST, direction=PadDirection.COL,
                                               size=(edge_size, dataset.num_edge_features), fill_value=0)
            self.graph_pad_op = PadHomoGraph(n_edge=edge_size, n_node=node_size, mode=PadMode.CONST)
        else:
            self.node_feat_pad_op = PadArray2d(dtype=np.float32, mode=PadMode.AUTO, direction=PadDirection.COL,
                                               fill_value=0)
            self.edge_feat_pad_op = PadArray2d(dtype=np.float32, mode=PadMode.AUTO, direction=PadDirection.COL,
                                               fill_value=0)
            self.graph_pad_op = PadHomoGraph(mode=PadMode.AUTO)
        # For Padding
        self.train_mask = np.array([True] * (self._batch_size + 1))
        self.train_mask[-1] = False

    def __getitem__(self, batch_graph_idx):
        graph_list = []
        feature_list = []
        edge_feat_list = []
        for idx in range(batch_graph_idx.shape[0]):
            graph_list.append(self._dataset[batch_graph_idx[idx]])
            feature_list.append(self._dataset.graph_feat(batch_graph_idx[idx]))
            edge_feat_list.append(self._dataset.graph_edge_feat(batch_graph_idx[idx]))
        # Batch Graph
        batch_graph = self.batch_fn(graph_list)
        # Pad Graph
        batch_graph = self.graph_pad_op(batch_graph)
        # Batch Node Feat
        batched_node_feat = np.concatenate(feature_list)
        # Pad NodeFeat
        batched_node_feat = self.node_feat_pad_op(batched_node_feat)
        batched_label = self._dataset.graph_label[batch_graph_idx]
        # Pad Label
        batched_label = np.append(batched_label, [batched_label[-1] * 0], axis=0)
        # Get Edge Feat
        batched_edge_feat = np.concatenate(edge_feat_list)
        batched_edge_feat = self.edge_feat_pad_op(batched_edge_feat)

        np_graph_mask = [1] * (self._batch_size + 1)
        np_graph_mask[-1] = 0
        constant_graph_mask = ms.Tensor(np_graph_mask, dtype=ms.int32)
        batchedgraphfiled = self.get_batched_graph_field(batch_graph, constant_graph_mask)
        row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask = batchedgraphfiled.get_batched_graph()
        return batched_label, batched_node_feat, batched_edge_feat, row, col, node_count, edge_count, node_map_idx,\
               edge_map_idx, graph_mask

    def get_batched_graph_field(self, batch_graph, constant_graph_mask):
        return BatchedGraphField(
            ms.Tensor.from_numpy(batch_graph.adj_coo[0]),
            ms.Tensor.from_numpy(batch_graph.adj_coo[1]),
            ms.Tensor(batch_graph.node_count, ms.int32),
            ms.Tensor(batch_graph.edge_count, ms.int32),
            ms.Tensor.from_numpy(batch_graph.batch_meta.node_map_idx),
            ms.Tensor.from_numpy(batch_graph.batch_meta.edge_map_idx),
            constant_graph_mask
        )

    def __len__(self):
        return self.length
