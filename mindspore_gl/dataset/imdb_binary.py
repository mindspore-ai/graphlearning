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
"""IMDBBinary"""
#pylint: disable=W0702
import random
from typing import Union
import os
import os.path as osp
import urllib.request
import zipfile
import numpy as np
from mindspore_gl.graph import MindHomoGraph
from .base_dataset import BaseDataSet


#pylint: disable=W0223
class IMDBBinary(BaseDataSet):

    """
    IMDBBinary Dataset, a source dataset for reading and parsing IMDBBinary dataset.

    About IMDBBinary dataset:

    IMDBBinary Dataset, a source dataset for reading and parsing IMDBBinary  dataset. IMDB-BINARY is a movie
    collaboration dataset that consists of the ego-networks of 1,000 actors/actresses who played roles in movies
    in IMDB. In each graph, nodes represent actors/actress, and there is an edge between them if they appear in the
    same movie. These graphs are derived from the Action and Romance genres.

    Statistics:

    - Nodes: 19773
    - Edges: 193062
    - Number of Graphs： 1000
    - Number of Classes: 2
    - Label split:

      - Train: 800
      - Valid: 200

    Dataset can be download here: <https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/IMDB-BINARY.zip>
    You can organize the dataset files into the following directory structure and read by `process` API.

    .. code-block::

        .
        ├── IMDB-BINARY_A.txt
        ├── IMDB-BINARY_graph_indicator.txt
        └── IMDB-BINARY_graph_labels.txt

    Args:
        root(str): path to the root directory that contains imdb_binary_with_mask.npz

    Raises:
        TypeError: if `root` is not a str.
        RuntimeError: if `root` does not contain data files.

    Examples:
        >>> from mindspore_gl.dataset.imdb_binary import IMDBBinary
        >>> root = "path/to/imdb_binary"
        >>> dataset = IMDBBinary(root)
    """
    url = 'https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/IMDB-BINARY.zip'
    def __init__(self, root):
        if not isinstance(root, str):
            raise TypeError(f"For '{self.cls_name}', the 'root' should be a str, "
                            f"but got {type(root)}.")
        self._root = root
        self._path = osp.join(root, 'imdb_binary_with_mask.npz')

        self._edge_array = None
        self._graphs = None

        self._node_feat = None
        self._graph_label = None
        self._graph_nodes = None
        self._graph_edges = None

        self._train_mask = None
        self._val_mask = None
        self._test_mask = None

        if osp.exists(self._path):
            self._load()
        else:
            path, file_name = self._download(self._root)
            self._process(path, file_name)
            self._load()

    def _download(self, save_dir):
        """Download dataset"""
        file = self.url.rpartition('/')[-1]
        path = osp.join(save_dir, file)
        unzip_name = file.rpartition('.')[0]
        unzip_path = osp.join(save_dir, unzip_name)
        if os.path.exists(unzip_path):
            return unzip_path, unzip_name
        data = urllib.request.urlopen(self.url)
        with open(path, 'wb') as f:
            while True:
                chunk = data.read(10*1024*1024)
                if not chunk:
                    break
                f.write(chunk)
        with zipfile.ZipFile(path, 'r') as f:
            f.extractall(save_dir)
        os.remove(path)
        return unzip_path, unzip_name

    def _process(self, path, file_name):
        """Process data"""
        label_file_name = file_name+'_graph_labels.txt'
        label_path = osp.join(path, label_file_name)
        self._graph_label = np.loadtxt(label_path)

        indicator_file_name = file_name+'_graph_indicator.txt'
        indicator_path = osp.join(path, indicator_file_name)
        graph_per_nodes = np.loadtxt(indicator_path, dtype=int)
        num_nodes = len(graph_per_nodes)
        self._graph_nodes = np.bincount(graph_per_nodes).cumsum().tolist()
        self._node_feat = np.zeros((num_nodes, 136))
        edges_file_name = file_name + '_A.txt'
        edges_path = osp.join(path, edges_file_name)
        load_edges = np.loadtxt(edges_path, delimiter=',', dtype=[('src', int), ('dst', int)])
        start = 0
        self._graph_edges = [0]
        adj_coo_row, adj_coo_col = [], []
        for i, node_count in enumerate(self._graph_nodes[1:]):
            for idx in range(start, len(load_edges)):
                if load_edges[idx][0] > node_count:
                    break
                elif idx == len(load_edges) - 1:
                    idx += 1
                    break
            adj_list = load_edges[start: idx].tolist()
            adj_list = list(set(adj_list))
            adj_list = sorted(adj_list, key=lambda x: [x[0], x[1]])
            src = [x[0] - 1 for x in adj_list]
            tag = [x[1] - 1 for x in adj_list]
            adj_coo_col += src
            adj_coo_row += tag
            last_edge = self._graph_edges[-1]
            self._graph_edges.append(last_edge + len(adj_list))
            start = idx
        mask_idx = list(range(len(self._graph_label)))
        random.shuffle(mask_idx)
        train_mask = [0] * len(mask_idx)
        for idx in mask_idx[len(mask_idx) // 10:]:
            train_mask[idx] = 1
        val_mask = [0] * len(mask_idx)
        for idx in mask_idx[:len(mask_idx) // 10]:
            val_mask[idx] = 1
        edge_array = np.array([adj_coo_col, adj_coo_row])

        for i in range(1, len(self._graph_nodes)):
            start = self._graph_nodes[i - 1]
            end = self._graph_nodes[i]
            for j in range(start, end):
                self._node_feat[j, j - start] = 1

        np.savez(self._path, edge_array=edge_array, train_mask=train_mask, val_mask=val_mask,
                 node_feat=self._node_feat, graph_label=self._graph_label,
                 graph_edges=self._graph_edges, graph_nodes=self._graph_nodes)

    def _load(self):
        """Load the saved npz dataset from files."""
        self._npz_file = np.load(self._path)
        self._edge_array = self._npz_file['edge_array'].astype(np.int32)
        self._graph_edges = self._npz_file['graph_edges'].astype(np.int32)
        self._graph_nodes = self._npz_file['graph_nodes'].astype(np.int32)
        self._graphs = np.array(list(range(len(self._graph_edges))))

    @property
    def node_feat_size(self):
        """
        Feature size of each node

        Returns:
            int, the number of feature size

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> node_feat_size = dataset.node_feat_size
        """
        return self.node_feat.shape[-1]

    @property
    def edge_feat_size(self):
        """
        Feature size of each edge

        Returns:
            int, the number of feature size

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> edge_feat_size = dataset.edge_feat_size
        """
        return 0

    @property
    def num_classes(self):
        """
        Number of label classes

        Returns:
            int, the number of classes

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> num_classes = dataset.num_classes
        """
        return len(np.unique(self.graph_label))

    @property
    def train_mask(self):
        """
        Mask of training nodes

        Returns:
            numpy.ndarray, array of mask

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> train_mask = dataset.train_mask
        """
        if self._train_mask is None:
            self._train_mask = self._npz_file['train_mask']
        return self._train_mask

    @property
    def val_mask(self):
        """
        Mask of validation nodes

        Returns:
            numpy.ndarray, array of mask

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
        Accumulative graph nodes count

        Returns:
            numpy.ndarray, array of accumulative nodes

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
        Accumulative graph edges count

        Returns:
            numpy.ndarray, array of accumulative edges

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
        Train graph id

        Returns:
            numpy.ndarray, array of train graph id

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> train_graphs = dataset.train_graphs
        """
        return (np.nonzero(self.train_mask)[0]).astype(np.int32)

    @property
    def val_graphs(self):
        """
        Valid graph id

        Returns:
            numpy.ndarray, array of valid graph id

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> val_graphs = dataset.val_graphs
        """
        return (np.nonzero(self.val_mask)[0]).astype(np.int32)

    @property
    def graph_count(self):
        """
        Total graph numbers

        Returns:
            int, numbers of graph

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> graph_count = dataset.graph_count
        """
        return len(self.graph_label)

    @property
    def node_feat(self):
        """
        Node features

        Returns:
            numpy.ndarray, array of node feature

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> node_feat = dataset.node_feat
        """
        if self._node_feat is None:
            self._node_feat = self._npz_file["node_feat"]
        return self._node_feat

    def graph_node_feat(self, graph_idx):
        """
        graph node features.

        Args:
            graph_idx (int): index of graph.

        Returns:
            - numpy.ndarray, node feature of graph.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> graph_node_feat = dataset.graph_node_feat(graph_idx)
        """
        return self.node_feat[self.graph_nodes[graph_idx]: self.graph_nodes[graph_idx + 1]]

    @property
    def graph_label(self):
        """
        Graph label

        Returns:
            numpy.ndarray, array of graph label

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> graph_label = dataset.graph_label
       """
        if self._graph_label is None:
            self._graph_label = self._npz_file["graph_label"]
        return self._graph_label.astype(np.int32)


    def __getitem__(self, idx) -> Union[MindHomoGraph, np.ndarray]:
        assert idx < self.graph_count, "Index out of range"
        res = MindHomoGraph()
        # reindex to 0
        coo_array = self._edge_array[:, self.graph_edges[idx]: self.graph_edges[idx + 1]] - self.graph_nodes[idx]
        res.set_topo_coo(coo_array)
        res.node_count = self.graph_nodes[idx + 1] - self.graph_nodes[idx]
        res.edge_count = self.graph_edges[idx + 1] - self.graph_edges[idx]
        return res
