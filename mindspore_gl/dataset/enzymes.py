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
"""Enzymes Dataset"""
#pylint: disable=W0702
import re
from typing import Union
import pathlib
import numpy as np
from mindspore_gl.graph import MindHomoGraph

class Enzymes:
    """
    Enzymes Dataset, a source dataset for reading and parsing Enzymes dataset.

    About Enzymes dataset:

    ENZYMES is a dataset of protein tertiary structures obtained from (Borgwardt et al., 2005) consisting of 600 enzymes
    from the BRENDA enzyme database (Schomburg et al., 2004). In this case the task is to correctly assign each enzyme
    to one of the 6 EC top-level classes.

    Statistics:

    - Graphs: 600
    - Nodes: 32.63
    - Edges: 62.14
    - Number of Classes: 6

    Dataset can be download here:
    `ENZYMES <https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/ENZYMES.zip>`_ .

    You can organize the dataset files into the following directory structure and read by `preprocess` API.

    .. code-block::

        .
        ├── ENZYMES_A.txt
        ├── ENZYMES_graph_indicator.txt
        ├── ENZYMES_graph_labels.txt
        ├── ENZYMES_node_attributes.txt
        ├── ENZYMES_node_labels.txt
        └── README.txt

    Args:
        root(str): path to the root directory that contains enzymes_with_mask.npz.

    Raises:
        TypeError: if `root` is not a str.
        RuntimeError: if `root` does not contain data files.

    Examples:
        >>> from mindspore_gl.dataset import Enzymes
        >>> root = "path/to/enzymes"
        >>> dataset = Enzymes(root)
    """

    dataset_url = ""

    def __init__(self, root):
        if not isinstance(root, str):
            raise TypeError(f"For '{self.cls_name}', the 'root' should be a str, "
                            f"but got {type(root)}.")

        self._root = pathlib.Path(root)
        self._path = self._root / 'enzymes_with_mask.npz'
        self._edge_array = None
        self._graphs = None

        self._node_feat = None
        self._graph_label = None
        self._graph_nodes = None
        self._graph_edges = None
        self._label_dim = None
        self._max_num_node = None

        self._train_mask = None
        self._val_mask = None
        self._test_mask = None

        if self._root.is_dir() and self._path.is_file():
            self._load()
        elif self._root.is_dir():
            self._preprocess()
            self._load()
        else:
            raise Exception('data file does not exist')

    def _preprocess(self):
        """process data"""
        adj_list, graph_indic, node_attrs, graph_labels, label_dim = self._get_info()
        max_node_nums = 0
        graph_edges, graph_nodes = [0], [0]
        adj_coo_row, adj_coo_col = [], []
        train_mask, test_mask, val_mask = [], [], []
        for i in range(1, 1 + len(adj_list)):
            src = [x[0] for x in adj_list[i]]
            tag = [x[1] for x in adj_list[i]]
            adj_coo_col += src
            adj_coo_row += tag
            last_edge = graph_edges[-1]
            graph_edges.append(last_edge + len(src))
            nodes = []
            for key in graph_indic:
                if graph_indic[key] == i:
                    nodes.append(key)
                elif graph_indic[key] > i:
                    break
            nodes_num = len(nodes)
            max_node_nums = max(max_node_nums, nodes_num)
            last_node = graph_nodes[-1]
            graph_nodes.append(last_node + nodes_num)
            if i % 10 == 0:
                test_mask.append(1)
                train_mask.append(0)
                val_mask.append(0)
            elif i % 9 == 0:
                val_mask.append(1)
                train_mask.append(0)
                test_mask.append(0)
            else:
                train_mask.append(1)
                test_mask.append(0)
                val_mask.append(0)

        edge_array = np.array([adj_coo_row, adj_coo_col])
        np.savez(self._path, edge_array=edge_array, train_mask=train_mask, val_mask=val_mask,
                 test_mask=test_mask, node_feat=node_attrs, graph_label=graph_labels, max_num_node=max_node_nums,
                 graph_edges=graph_edges, graph_nodes=graph_nodes, label_dim=label_dim)

    def _get_info(self):
        """get graphs info"""
        filename_graph_indic = self._root / 'ENZYMES_graph_indicator.txt'
        graph_indic = {}
        with open(filename_graph_indic) as f:
            i = 1
            for line in f:
                line = line.strip("\n")
                graph_indic[i] = int(line)
                i += 1

        filename_node_attrs = self._root / 'ENZYMES_node_attributes.txt'
        node_attrs = []
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip(r"\s\n")
                attrs = [float(attr) for attr in re.split(r"[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
        node_attrs = np.array(node_attrs)

        filename_graphs = self._root / 'ENZYMES_graph_labels.txt'
        graph_labels = []
        label_vals = []
        with open(filename_graphs) as f:
            for line in f:
                line = line.strip("\n")
                val = int(line)
                if val not in label_vals:
                    label_vals.append(val)
                graph_labels.append(val)
        label_map_to_int = {val: i for i, val in enumerate(label_vals)}
        graph_labels = np.array([label_map_to_int[l] for l in graph_labels])
        label_dim = len(set(graph_labels))

        filename_adj = self._root / 'ENZYMES_A.txt'
        adj_list = {i: [] for i in range(1, len(graph_labels) + 1)}
        index_graph = {i: [] for i in range(1, len(graph_labels) + 1)}
        num_edges = 0
        with open(filename_adj) as f:
            for line in f:
                line = line.strip("\n").split(",")
                e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
                adj_list[graph_indic[e0]].append((e0, e1))
                index_graph[graph_indic[e0]] += [e0, e1]
                num_edges += 1
        for k in index_graph.keys():
            index_graph[k] = [u - 1 for u in set(index_graph[k])]
        return adj_list, graph_indic, node_attrs, graph_labels, label_dim

    def _load(self):
        """Load the saved npz dataset from files."""
        self._npz_file = np.load(self._path)
        self._edge_array = self._npz_file['edge_array']
        self._graph_edges = self._npz_file['graph_edges']
        self._graphs = np.array(list(range(len(self._graph_edges))))

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
    def label_dim(self):
        """
        Number of label classes.

        Returns:
            - int, the number of classes.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> label_dim = dataset.label_dim
        """
        if self._label_dim is None:
            self._label_dim = self._npz_file['label_dim']
        return int(self._label_dim)

    @property
    def max_num_node(self):
        """
        Max number of nodes in one graph.

        Returns:
            - int, the max number of node number.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> max_num_node = dataset.max_num_node
        """
        if self._max_num_node is None:
            self._max_num_node = self._npz_file['max_num_node']
        return int(self._max_num_node)

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
        if self._graph_nodes is None:
            self._graph_nodes = self._npz_file['graph_nodes']
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
            self._graph_edges = self._npz_file['graph_edges']
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
        return np.nonzero(self.train_mask)[0]

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
        return np.nonzero(self.val_mask)[0]

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
        return np.nonzero(self.test_mask)[0]

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
        return len(self._graphs)

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

    def graph_feat(self, graph_idx):
        """
        graph features.

        Args:
            graph_idx (int): index of graph.

        Returns:
            - numpy.ndarray, node feature of graph.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> graph_feat = dataset.graph_feat(graph_idx)
        """
        return self.node_feat[self.graph_nodes[graph_idx]: self.graph_nodes[graph_idx + 1]]

    @property
    def graph_label(self):
        """
        Graph label.

        Returns:
            - numpy.ndarray, array of graph label.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> node_feat = dataset.graph_label
       """
        if self._graph_label is None:
            self._graph_label = self._npz_file["graph_label"]
        return self._graph_label

    def __getitem__(self, idx) -> Union[MindHomoGraph, np.ndarray]:
        assert idx < self.graph_count, "Index out of range"
        res = MindHomoGraph()
        # reindex to 0
        coo_array = self._edge_array[:, self.graph_edges[idx]: self.graph_edges[idx + 1]] - (self.graph_nodes[idx] + 1)
        res.set_topo_coo(coo_array)
        res.node_count = self.graph_nodes[idx + 1] - self.graph_nodes[idx]
        res.edge_count = self.graph_edges[idx + 1] - self.graph_edges[idx]
        return res
