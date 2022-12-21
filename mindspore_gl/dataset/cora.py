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
"""CoraV2"""
import os
import pickle as pkl
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix
from mindspore_gl.graph.graph import MindHomoGraph, CsrAdj


class CoraV2:
    r"""
    Cora Dataset, a source dataset for reading and parsing Cora dataset.

    Args:
        root(str): path to the root directory that contains cora_v2_with_mask.npz.
        name(str): select dataset type, optional: ["cora_v2", "citeseer", "pubmed"].
    Raises:
        RuntimeError: If root does not contain data files.

    Examples:
        >>> from mindspore_gl.dataset import CoraV2
        >>> root = "path/to/cora_v2_with_mask.npz"
        >>> dataset = CoraV2(root)

    About Cora dataset:

    The Cora dataset consists of 2708 scientific publications classified into one of seven classes. The citation network
    consists of 10556 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the
    absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words.

    Cora_v2 Statistics:

    - Nodes: 2708
    - Edges: 10556
    - Number of Classes: 7
    - Label split:

      - Train: 140
      - Valid: 500
      - Test: 1000

    Dataset can be download here:

    `cora_v2 <https://data.dgl.ai/dataset/cora_v2.zip>`_

    `citeseer <https://data.dgl.ai/dataset/citeseer.zip>`_

    `pubmed <https://data.dgl.ai/dataset/pubmed.zip>`_

    You can organize the dataset files into the following directory structure and read by `process` API.

    .. code-block::

        .
        └── corav2
            ├── ind.cora_v2.allx
            ├── ind.cora_v2.ally
            ├── ind.cora_v2.graph
            ├── ind.cora_v2.test.index
            ├── ind.cora_v2.tx
            ├── ind.cora_v2.ty
            ├── ind.cora_v2.x
            └── ind.cora_v2.y
    """

    def __init__(self, root, name='cora_v2'):
        if not isinstance(root, str):
            raise TypeError(f"For '{self.cls_name}', the 'root' should be a str, "
                            f"but got {type(root)}.")
        self._root = root
        self._name = name
        self._path = os.path.join(root, self._name+'_with_mask.npz')

        self._csr_row = None
        self._csr_col = None
        self._nodes = None

        self._node_feat = None
        self._node_label = None

        self._train_mask = None
        self._val_mask = None
        self._test_mask = None
        self._npz_file = None

        if os.path.exists(self._path) and os.path.isfile(self._path):
            self._load()
        elif os.path.exists(self._root):
            self._preprocess()
            self._load()
        else:
            raise Exception('data file does not exist')

    def _preprocess(self):
        """Download and process data"""
        names = ['y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        dataset_str = self._name

        for name in names:
            try:
                with open("{}/ind.{}.{}".format(self._root, dataset_str, name), 'rb') as f:
                    objects.append(pkl.load(f, encoding='latin1'))
            except IOError as e:
                raise e

        y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = _parse_index_file("{}/ind.{}.test.index".format(self._root, dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if self._name == 'citeseer':
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), tx.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        features = _normalize_cora_features(features)
        graph = nx.Graph(nx.from_dict_of_lists(graph))
        graph = graph.to_directed()

        onehot_labels = np.vstack((ally, ty))
        onehot_labels[test_idx_reorder, :] = onehot_labels[test_idx_range, :]
        labels = np.argmax(onehot_labels, 1)

        adj_coo_row = []
        adj_coo_col = []
        line_count = 0

        for e in graph.edges:
            adj_coo_row.append(e[1])
            adj_coo_col.append(e[0])
            line_count += 1

        for i in range(len(labels)):
            adj_coo_row.append(i)
            adj_coo_col.append(i)

        num_nodes = len(labels)
        num_edges = len(adj_coo_row)
        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        train_mask = _sample_mask(idx_train, num_nodes)
        val_mask = _sample_mask(idx_val, num_nodes)
        test_mask = _sample_mask(idx_test, num_nodes)

        adj_coo_matrix = coo_matrix((np.ones(len(adj_coo_row), dtype=bool), (adj_coo_row, adj_coo_col)),
                                    shape=(num_nodes, num_nodes))
        out_degrees = np.sum(adj_coo_matrix, axis=1)
        out_degrees = np.ravel(out_degrees)
        in_degrees = np.sum(adj_coo_matrix, axis=0)
        in_degrees = np.ravel(in_degrees)
        adj_csr_matrix = adj_coo_matrix.tocsr()
        features = np.array(features, np.float32)
        np.savez(self._path, feat=features, label=labels, test_mask=test_mask,
                 train_mask=train_mask, val_mask=val_mask, adj_coo_row=adj_coo_row, adj_coo_col=adj_coo_col,
                 adj_csr_indptr=adj_csr_matrix.indptr, adj_csr_indices=adj_csr_matrix.indices, in_degrees=in_degrees,
                 out_degrees=out_degrees, adj_csr_data=adj_csr_matrix.data,
                 n_edges=num_edges, n_nodes=num_nodes, n_classes=onehot_labels.shape[1])

    def _load(self):
        """Load the saved npz dataset from files."""
        self._npz_file = np.load(self._path)
        self._csr_row = self._npz_file['adj_csr_indptr'].astype(np.int32)
        self._csr_col = self._npz_file['adj_csr_indices'].astype(np.int32)

        self._nodes = np.array(list(range(len(self._csr_row) - 1)))

    @property
    def num_features(self):
        """
        Feature size of each node

        Returns:
            - int, the number of feature size

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> num_features = dataset.num_features
        """
        return self.node_feat.shape[1]

    @property
    def num_classes(self):
        """
        Number of label classes

        Returns:
            - int, the number of classes

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> num_classes = dataset.num_classes
        """
        return len(np.unique(self.node_label))

    @property
    def train_mask(self):
        """
        Mask of training nodes

        Returns:
            - numpy.ndarray, array of mask

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
        Mask of test nodes

        Returns:
            - numpy.ndarray, array of mask

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
        Mask of validation nodes

        Returns:
            - numpy.ndarray, array of mask

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> val_mask = dataset.val_mask
        """
        if self._val_mask is None:
            self._val_mask = self._npz_file['val_mask']
        return self._val_mask

    @property
    def train_nodes(self):
        """
        training nodes indexes

        Returns:
            - numpy.ndarray, array of training nodes

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> train_nodes = dataset.train_nodes
        """
        return (np.nonzero(self.train_mask)[0]).astype(np.int32)

    @property
    def node_count(self):
        """
        Number of nodes

        Returns:
            - int, length of csr row

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> node_count = dataset.node_count
        """
        return len(self._csr_row)

    @property
    def edge_count(self):
        """
        Number of edges

        Returns:
            - int, length of csr col

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> edge_count = dataset.edge_count
        """
        return len(self._csr_col)

    @property
    def node_feat(self):
        """
        Node features

        Returns:
            - numpy.ndarray, array of node feature

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> node_feat = dataset.node_feat
        """
        if self._node_feat is None:
            self._node_feat = self._npz_file["feat"]

        return self._node_feat

    @property
    def node_label(self):
        """
        Ground truth labels of each node

        Returns:
            - numpy.ndarray, array of node label

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> node_label = dataset.node_label
        """
        if self._node_label is None:
            self._node_label = self._npz_file["label"]
        return self._node_label.astype(np.int32)

    @property
    def adj_coo(self):
        """
        Return the adjacency matrix of COO representation

        Returns:
            - numpy.ndarray, array of coo matrix.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> node_label = dataset.adj_coo
        """
        return csr_matrix((np.ones(self._csr_col.shape), self._csr_col, self._csr_row)).tocoo(copy=False)

    @property
    def adj_csr(self):
        """
        Return the adjacency matrix of CSR representation.

        Returns:
            - numpy.ndarray, array of csr matrix.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> node_label = dataset.adj_csr
        """
        return csr_matrix((np.ones(self._csr_col.shape), self._csr_col, self._csr_row))

    def __getitem__(self, idx):
        assert idx == 0, "Cora only has one graph"
        graph = MindHomoGraph()
        node_dict = {idx: idx for idx in range(self.node_count)}
        edge_ids = np.array(list(range(self.edge_count))).astype(np.int32)
        graph.set_topo(CsrAdj(self._csr_row, self._csr_col), node_dict=node_dict, edge_ids=edge_ids)
        return graph


def _parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def _normalize_cora_features(features):
    row_sum = np.array(features.sum(1))
    r_inv = np.power(row_sum * 1.0, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return np.asarray(features.todense())


def _sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l, dtype=bool)
    mask[idx] = True
    return mask
