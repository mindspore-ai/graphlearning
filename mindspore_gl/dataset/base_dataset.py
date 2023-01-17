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
"""Define the base dataset for graph datasets."""


class BaseDataSet:
    """Base dataset"""

    @property
    def train_mask(self):
        raise NotImplementedError()

    @property
    def test_mask(self):
        raise NotImplementedError()

    @property
    def val_mask(self):
        raise NotImplementedError()

    @property
    def node_count(self):
        raise NotImplementedError()

    @property
    def edge_count(self):
        raise NotImplementedError()

    @property
    def node_feat(self):
        raise NotImplementedError()

    @property
    def edge_feat(self):
        raise NotImplementedError()

    @property
    def node_feat_size(self):
        raise NotImplementedError()

    @property
    def edge_feat_size(self):
        raise NotImplementedError()

    @property
    def node_label(self):
        raise NotImplementedError()

    @property
    def edge_label(self):
        raise NotImplementedError()

    @property
    def num_classes(self):
        raise NotImplementedError()

    @property
    def adj_coo(self):
        raise NotImplementedError()

    @property
    def adj_csr(self):
        raise NotImplementedError()

    @property
    def train_nodes(self):
        raise NotImplementedError()

    @property
    def test_nodes(self):
        raise NotImplementedError()

    @property
    def val_nodes(self):
        raise NotImplementedError()

    @property
    def train_graphs(self):
        raise NotImplementedError()

    @property
    def test_graphs(self):
        raise NotImplementedError()

    @property
    def val_graphs(self):
        raise NotImplementedError()

    @property
    def graph_nodes(self):
        raise NotImplementedError()

    @property
    def graph_edges(self):
        raise NotImplementedError()

    @property
    def graph_count(self):
        raise NotImplementedError()

    @property
    def graph_label(self):
        raise NotImplementedError()

    def graph_node_feat(self, graph_idx):
        raise NotImplementedError()

    def graph_edge_feat(self, graph_idx):
        raise NotImplementedError()
