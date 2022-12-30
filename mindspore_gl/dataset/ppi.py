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
import os
import os.path as osp
from typing import Union
import json
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from mindspore_gl.graph import MindHomoGraph

class PPI:
    """
    PPI  Dataset, a source dataset for reading and parsing PPI dataset.

    About PPI dataset:

    protein roles—in terms of their cellular functions from gene ontology—in various protein-protein interaction (PPI)
    graphs, with each graph corresponding to a different human tissue. positional gene sets are used, motif gene sets
    and immunological signatures as features and gene ontology sets as labels (121 in total), collected from the
    Molecular Signatures Database. The average graph contains 2373 nodes, with an average degree of 28.8.

    Statistics:

    - Graphs: 24
    - Nodes: ~2245.3
    - Edges: ~61,318.4
    - Number of Classes: 121
    - Label split:

      - Train examples: 20
      - Valid examples: 2
      - Test examples: 2

    Dataset can be download here: <https://data.dgl.ai/dataset/ppi.zip>

    You can organize the dataset files into the following directory structure and read by `preprocess` API.

    .. code-block::

        .
        └── ppi
            ├── valid_feats.npy
            ├── valid_labels.npy
            ├── valid_graph_id.npy
            ├── valid_graph.json
            ├── train_feats.npy
            ├── train_labels.npy
            ├── train_graph_id.npy
            ├── train_graph.json
            ├── test_feats.npy
            ├── test_labels.npy
            ├── test_graph_id.npy
            └── test_graph.json

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

    def __init__(self, root):
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
        self._graph_label = None

        if os.path.exists(self._path) and os.path.isfile(self._path):
            self._load()
        elif os.path.exists(self._root):
            self._preprocess()
            self._load()
        else:
            raise Exception('data file does not exist')
        self._load()

    def _preprocess(self):
        """process data"""
        node_feat = np.empty((1, 50))
        node_label = np.empty((1, 121))
        node_nums = []
        adj_coo_row, adj_coo_col = [], []
        graph_edge = [0]
        graph_node = [0]
        mode_node = 0
        for mode in ['train', 'test', 'valid']:
            graph_file = os.path.join(self._root, '{}_graph.json'.format(mode))
            label_file = os.path.join(self._root, '{}_labels.npy'.format(mode))
            feat_file = os.path.join(self._root, '{}_feats.npy'.format(mode))
            graph_id_file = os.path.join(self._root, '{}_graph_id.npy'.format(mode))

            g_data = json.load(open(graph_file))
            node_labels = np.load(label_file)
            node_feats = np.load(feat_file)
            graph_id = np.load(graph_id_file)
            lo, hi = 1, 21
            if mode == 'valid':
                lo, hi = 21, 23
            elif mode == 'test':
                lo, hi = 23, 25
            for g_id in range(lo, hi):
                g_mask = np.where(graph_id == g_id)[0]
                feature = node_feats[g_mask]
                labels = node_labels[g_mask]
                n_nodes = len(g_mask)
                node_feat = np.concatenate((node_feat, feature), axis=0)
                node_label = np.concatenate((node_label, labels), axis=0)
                node_nums.append(n_nodes)
                last_node = graph_node[-1]
                graph_node.append(last_node + n_nodes)
                sub_graph = {"directed": False, "multigraph": False, "graph": {}}
                node_id = []
                for n in g_mask:
                    node_id.append({"id": n})
                sub_graph['nodes'] = node_id
                sub_graph_link = []
                for link in g_data['links']:
                    if link['source'] in g_mask and link['target'] in g_mask:
                        sub_graph_link.append(link)
                last_edge = graph_edge[-1]
                graph_edge.append(last_edge + len(sub_graph_link))
                sub_graph['links'] = sub_graph_link
                sub_graph = nx.DiGraph(json_graph.node_link_graph(sub_graph))
                sub_row = []
                sub_col = []
                for e in sub_graph.edges:
                    sub_row.append(e[0])
                    sub_col.append(e[1])
                sub_row = [i + mode_node for i in sub_row]
                sub_col = [i + mode_node for i in sub_col]
                adj_coo_row += sub_row
                adj_coo_col += sub_col
            mode_node = graph_node[-1]
        node_feat = node_feat[1:, :]
        node_label = node_label[1:, :]
        edge_array = np.array([adj_coo_row, adj_coo_col])
        save_path = os.path.join(self._root, 'ppi_with_mask.npz')
        train_mask = [1] * 20 + [0] * 4
        val_mask = [0] * 20 + [1] * 2 + [0] * 2
        test_mask = [0] * 22 + [1] * 2
        np.savez(save_path, edge_array=edge_array, train_mask=train_mask, val_mask=val_mask,
                 test_mask=test_mask, node_feat=node_feat, node_label=node_label, node_nums=node_nums,
                 graph_edges=graph_edge, graph_nodes=graph_node)

    def _load(self):
        """Load the saved npz dataset from files."""
        self._npz_file = np.load(self._path, allow_pickle=True)
        self._edge_array = self._npz_file['edge_array'].astype(np.int32)
        self._graph_edges = self._npz_file['graph_edges'].astype(np.int32)

        self._train_mask = self._npz_file['train_mask']
        self._test_mask = self._npz_file['test_mask']
        self._val_mask = self._npz_file['val_mask']
        self._edge_array = self._npz_file['edge_array']
        self._node_feat = self._npz_file['node_feat']
        self._node_label = self._npz_file['node_label']

    @property
    def num_features(self):
        """
        Feature size of each node.

        Returns:
            - int, the number of feature size.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> num_features = dataset.num_features
        """
        return self.node_feat.shape[-1]

    @property
    def num_classes(self):
        """
        Number of label classes.

        Returns:
            - int, the number of classes.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> num_classes = dataset.num_classes
        """
        return self.node_label.shape[-1]

    @property
    def train_mask(self):
        """
        Mask of training nodes.

        Returns:
            - numpy.ndarray, array of mask.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> train_mask = dataset.train_mask
        """
        if self._train_mask is None:
            self._train_mask = self._npz_file['train_mask']
        return self._train_mask

    @property
    def test_mask(self):
        """
       Mask of test nodes.

       Returns:
           - numpy.ndarray, array of mask.

       Examples:
           >>> #dataset is an instance object of Dataset
           >>> test_mask = dataset.test_mask
       """
        if self._test_mask is None:
            self._test_mask = self._npz_file['test_mask']
        return self._test_mask

    @property
    def val_mask(self):
        """
        Mask of validation nodes.

        Returns:
            - numpy.ndarray, array of mask.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> val_mask = dataset.val_mask
        """
        if self._val_mask is None:
            self._val_mask = self._npz_file['val_mask']
        return self._val_mask

    @property
    def graph_nodes(self):
        """
        Accumulative graph nodes count.

        Returns:
            - numpy.ndarray, array of accumulative nodes.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> val_mask = dataset.graph_nodes
        """
        try:
            if self._graph_nodes is None:
                self._graph_nodes = self._npz_file['graph_nodes'].astype(np.int32)
            return self._graph_nodes
        except:
            self.load()
            self._graph_nodes = self._npz_file['graph_nodes'].astype(np.int32)
            return self._graph_nodes

    @property
    def graph_edges(self):
        """
        Accumulative graph edges count.

        Returns:
            - numpy.ndarray, array of accumulative edges.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> val_mask = dataset.graph_edges
        """
        if self._graph_edges is None:
            self._graph_edges = self._npz_file['graph_edges'].astype(np.int32)
        return self._graph_edges

    @property
    def train_graphs(self):
        """
        Train graph id.

        Returns:
            - numpy.ndarray, array of train graph id.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> train_graphs = dataset.train_graphs
        """
        return (np.nonzero(self.train_mask)[0]).astype(np.int32)

    @property
    def val_graphs(self):
        """
        Valid graph id.

        Returns:
            - numpy.ndarray, array of valid graph id.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> val_graphs = dataset.val_graphs
        """
        return (np.nonzero(self.val_mask)[0]).astype(np.int32)

    @property
    def test_graphs(self):
        """
        Test graph id.

        Returns:
            - numpy.ndarray, array of test graph id.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> test_graphs = dataset.test_graphs
        """
        return (np.nonzero(self.test_mask)[0]).astype(np.int32)

    @property
    def graph_count(self):
        """
        Total graph numbers.

        Returns:
            - int, numbers of graph.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> node_feat = dataset.node_feat
        """
        return len(self.train_mask)

    @property
    def node_feat(self):
        """
        Node features.

        Returns:
            - numpy.ndarray, array of node feature.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> node_feat = dataset.node_feat
        """
        if self._node_feat is None:
            self._node_feat = self._npz_file["node_feat"]

        return self._node_feat

    @property
    def node_label(self):
        """
        Ground truth labels of each node.

        Returns:
            - numpy.ndarray, array of node label.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> node_label = dataset.node_label
        """
        if self._node_label is None:
            self._node_label = self._npz_file["node_label"]

        return self._node_label

    def graph_feat(self, graph_idx):
        """
        graph features.

        Args:
            graph_idx (int): index of graph.

        Returns:
            - numpy.ndarray, node feature of graph.
        """
        return self.node_feat[self.graph_nodes[graph_idx]: self.graph_nodes[graph_idx + 1]]

    def graph_label(self, graph_idx):
        """
        graph label.

        Args:
            graph_idx (int): index of graph.

        Returns:
            - numpy.ndarray, node label of graph.
        """
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
