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
"""BlogCatalog Dataset"""
import os
import os.path as osp
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from mindspore_gl.graph import MindHomoGraph, CsrAdj
from .base_dataset import BaseDataSet

#pylint: disable=W0223
class BlogCatalog(BaseDataSet):
    """
    BlogCatalog Dataset, a source dataset for reading and parsing BlogCatalog dataset.

    About BlogCatalog dataset:

    This is the data set crawled from BlogCatalog. BlogCatalog is a social blog
    directory website. This contains the friendship network crawled and group memberships. For easier understanding,
    all the contents are organized in CSV file format.

    Statistics:

    - Nodes: 10,312
    - Edges: 333,983
    - Number of Classes: 39

    Dataset can be download here: `BlogCatalog <https://figshare.com/articles/dataset/BlogCatalog_dataset/11923611>`_ .

    You can organize the dataset files into the following directory structure and read by `preprocess` API.

    .. code-block::

        .
        └── ppi
            ├── edges.csv
            ├── group-edges.csv
            ├── groups.csv
            └── nodes.csv

    Args:
        root(str): path to the root directory that contains blog_catalog.npz.

    Raises:
        TypeError: if `root` is not a str.
        RuntimeError: if `root` does not contain data files.

    Examples:
        >>> from mindspore_gl.dataset.blog_catalog import BlogCatalog
        >>> root = "path/to/blog_catalog"
        >>> dataset = BlogCatalog(root)
    """
    def __init__(self, root):
        if not isinstance(root, str):
            raise TypeError(f"For '{self.cls_name}', the 'root' should be a str, "
                            f"but got {type(root)}.")
        self._root = root
        self._path = osp.join(root, 'blog_catalog.npz')

        self._csr_row = None
        self._csr_col = None
        self._nodes = None

        self._vocab = None
        self._node_label = None

        self._npz_file = None

        if os.path.exists(self._path) and os.path.isfile(self._path):
            self._load()
        elif os.path.exists(self._root):
            self._preprocess()
            self._load()
        else:
            raise Exception('data file does not exist')

    def _preprocess(self):
        """Process data"""
        nodes = pd.read_csv(ops.join(self._root, 'nodes.csv'), header=None)
        nodes = list(nodes.values[:, 0])
        node_num = len(nodes)
        groups = pd.read_csv(ops.join(self._root, 'groups.csv'), header=None)
        groups = list(groups.values[:, 0])
        edges = pd.read_csv(ops.join(self._root, 'edges.csv'), header=None)
        group_edges = pd.read_csv(ops.join(self._root, 'group-edges.csv'), header=None)
        group_edges = group_edges.drop_duplicates(subset=[0])
        vocab = group_edges.values[:, 0] - 1
        label = group_edges.values[:, 1]
        edges = edges.values
        dir_row = edges[:, 0] - 1
        dir_col = edges[:, 1] - 1
        row = np.hstack((dir_row, dir_col))
        col = np.hstack((dir_col, dir_row))
        data = [1] * len(row)
        coo = coo_matrix((data, (row, col)), shape=(node_num, node_num))
        crs = coo.tocsr()
        indptr = crs.indptr
        indces = crs.indices
        np.savez(self._path, num_classes=len(groups), adj_csr_indptr=indptr,
                 adj_csr_indices=indces, label=label, vocab=vocab)

    def _load(self):
        """Load the saved npz dataset from files."""
        self._npz_file = np.load(self._path)
        self._csr_row = self._npz_file['adj_csr_indptr'].astype(np.int32)
        self._csr_col = self._npz_file['adj_csr_indices'].astype(np.int32)
        self._nodes = np.array(list(range(len(self._csr_row) - 1)))

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
        return int(self._npz_file["num_classes"])

    @property
    def node_count(self):
        """
        Number of nodes.

        Returns:
            - int, length of csr row.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> node_count = dataset.node_count
        """
        return len(self._csr_row) - 1

    @property
    def edge_count(self):
        """
        Number of edges.

        Returns:
            - int, length of csr col.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> edge_count = dataset.edge_count
        """
        return len(self._csr_col)

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
            self._node_label = self._npz_file["label"]
        return self._node_label.astype(np.int32)

    @property
    def vocab(self):
        """
        ID of each node.

        Returns:
            - numpy.ndarray, array of node id.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> node_label = dataset.vocab
        """
        if self._vocab is None:
            self._vocab = self._npz_file["vocab"]
        return self._vocab.astype(np.int32)

    @property
    def adj_coo(self):
        """
        Return the adjacency matrix of COO representation.

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
        assert idx == 0, "Blog Catalog only has one graph"
        graph = MindHomoGraph()
        node_dict = {idx: idx for idx in range(self.node_count)}
        edge_ids = np.array(list(range(self.edge_count))).astype(np.int32)
        graph.set_topo(CsrAdj(self._csr_row, self._csr_col), node_dict=node_dict, edge_ids=edge_ids)
        return graph
