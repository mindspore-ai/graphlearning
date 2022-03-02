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
"""PPI Dataset"""
#pylint: disable=W0702
from typing import Optional, Union
import os.path as osp
import numpy as np
from mindspore_gl.graph.graph import MindHomoGraph


class PPI:
    """
    PPI  Dataset, a source dataset for reading and parsing PPI dataset.

    Args:
        root(str): path to the root directory that contains ppi_with_mask.npz.

    Raises:
        TypeError: if `root` is not a str.
        RuntimeError: if `root` does not contain data files.

    Examples:
        >>> from mindspore_gl.dataset.ppi import PPI
        >>> root = "path/to/ppi"
        >>> dataset = PPI(root)

    """

    def __init__(self, root: Optional[str] = None):
        if not isinstance(root, str):
            raise TypeError(f"For '{self.cls_name}', the 'root' should be a str, "
                            f"but got {type(root)}.")
        self._root = root
        self._path = osp.join(root, 'ppi_with_mask.npz')

        self._edge_array = None
        self._graphs = None

        self._node_feat = None
        self._node_label = None
        self._graph_nodes = None
        self._graph_edges = None

        self._train_mask = None
        self._val_mask = None
        self._test_mask = None

        self.load()

    def load(self):
        self._npz_file = np.load(self._path)
        self._edge_array = self._npz_file['edge_array'].astype(np.int32)
        self._graph_edges = self._npz_file['graph_edges'].astype(np.int32)

        self._graphs = np.array(list(range(len(self._graph_edges))))

    @property
    def num_features(self):
        return self.node_feat.shape[-1]

    @property
    def num_classes(self):
        return self.node_label.shape[-1]

    @property
    def train_mask(self):
        if self._train_mask is None:
            self._train_mask = self._npz_file['train_mask']
        return self._train_mask

    @property
    def test_mask(self):
        if self._test_mask is None:
            self._test_mask = self._npz_file['test_mask']
        return self._test_mask

    @property
    def val_mask(self):
        if self._val_mask is None:
            self._val_mask = self._npz_file['val_mask']
        return self._val_mask

    @property
    def graph_nodes(self):
        """graph nodes"""
        try:
            if self._graph_nodes is None:
                self._graph_nodes = self._npz_file['graph_nodes'].astype(np.int32)
            return self._graph_nodes
        except:
            ############################################################################
            # NPZ File Is Known To Be Bugged When Sharing NPZFileHandler Across Process
            # Checkout This Issue: https://github.com/numpy/numpy/issues/18124
            #############################################################################
            self.load()
            self._graph_nodes = self._npz_file['graph_nodes'].astype(np.int32)
            return self._graph_nodes

    @property
    def graph_edges(self):
        if self._graph_edges is None:
            self._graph_edges = self._npz_file['graph_edges'].astype(np.int32)
        return self._graph_edges

    @property
    def train_graphs(self):
        return (np.nonzero(self.train_mask)[0]).astype(np.int32)

    @property
    def val_graphs(self):
        return (np.nonzero(self.val_mask)[0]).astype(np.int32)

    @property
    def test_graphs(self):
        return (np.nonzero(self.test_mask)[0]).astype(np.int32)

    @property
    def graph_count(self):
        return len(self._graphs)

    @property
    def node_feat(self):
        if self._node_feat is None:
            self._node_feat = self._npz_file["node_feat"]

        return self._node_feat

    @property
    def node_label(self):
        if self._node_label is None:
            self._node_label = self._npz_file["node_label"]

        return self._node_label

    def graph_feat(self, graph_idx):
        return self.node_feat[self.graph_nodes[graph_idx]: self.graph_nodes[graph_idx + 1]]

    def graph_label(self, graph_idx):
        return self.node_label[self.graph_nodes[graph_idx]: self.graph_nodes[graph_idx + 1]]

    def __getitem__(self, idx) -> Union[MindHomoGraph, np.ndarray]:
        assert idx < self.graph_count, "Index out of range"
        res = MindHomoGraph()
        # reindex to 0
        coo_array = self._edge_array[:, self.graph_edges[idx]: self.graph_edges[idx + 1]] - self.graph_nodes[idx]
        res.set_topo_coo(coo_array)
        res.node_count = self.graph_nodes[idx + 1] - self.graph_nodes[idx]
        res.edge_count = self.graph_edges[idx + 1] - self.graph_edges[idx]
        return res
