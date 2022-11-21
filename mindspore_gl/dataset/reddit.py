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
"""Reddit Dataset"""
#pylint: disable=W0702

from pathlib import Path
import numpy as np
import scipy.sparse as sp
from mindspore_gl.graph import MindHomoGraph, CsrAdj

class Reddit:
    """
    Reddit Dataset, a source dataset for reading and parsing Reddit dataset.

    Args:
        root(str): path to the root directory that contains reddit_with_mask.npz

    Raises:
        TypeError: if `root` is not a str.
        RuntimeError: if `root` does not contain data files.

    Examples:
        >>> from mindspore_gl.dataset import Reddit
        >>> root = "path/to/reddit"
        >>> dataset = Reddit(root)

    About Reddit dataset:

    The node label in this case is the community, or “subreddit”, that a post belongs to.
    The authors sampled 50 large communities and built a post-to-post graph, connecting
    posts if the same user comments on both. In total this dataset contains 232,965
    posts with an average degree of 492. We use the first 20 days for training and the
    remaining days for testing (with 30% used for validation).

    Statistics:

    - Nodes: 232,965
    - Edges: 114,615,892
    - Number of classes: 41

    Dataset can be download here: <https://data.dgl.ai/dataset/reddit.zip>
    You can organize the dataset files into the following directory structure and read by `preprocess` API.

    .. code-block::
        .
        ├── reddit_data.npz
        ├── reddit_graph.npz
        └── reddit_smaller.npz

    """

    def __init__(self, root):
        if not isinstance(root, str):
            raise TypeError(f"For '{self.cls_name}', the 'root' should be a str, "
                            f"but got {type(root)}.")
        self._root = Path(root)
        self._path = self._root / 'reddit_with_mask.npz'

        self._csr_row = None
        self._csr_col = None
        self._nodes = None

        self._node_feat = None
        self._node_label = None

        self._train_mask = None
        self._val_mask = None
        self._test_mask = None
        self._npz_file = None

        if self._path.is_file():
            self.load()
        elif self._root.is_dir():
            self.preprocess()
            self.load()
        else:
            raise Exception('data file does not exist')

    def preprocess(self):
        """process data"""
        coo_adj = sp.load_npz(self._root / "reddit_graph.npz")
        csr = coo_adj.tocsr()
        indices = csr.indices
        indptr = csr.indptr
        reddit_data = np.load(self._root / "reddit_data.npz")
        features = np.array(reddit_data["feature"], np.float32)
        labels = reddit_data["label"]
        node_types = reddit_data["node_types"]
        train_mask = (node_types == 1)
        val_mask = (node_types == 2)
        test_mask = (node_types == 3)
        np.savez(self._path, adj_csr_indptr=indptr, adj_csr_indices=indices, feat=features, label=labels,
                 train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    def load(self):
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
            int, the number of feature size

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
            int, the number of classes

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
            numpy.ndarray, array of mask

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
            numpy.ndarray, array of mask

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
            numpy.ndarray, array of mask

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
            numpy.ndarray, array of training nodes

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> train_nodes = dataset.train_nodes
        """
        return (np.nonzero(self.train_mask)[0]).astype(np.int32)

    @property
    def val_nodes(self):
        """
        Val nodes indexes

        Returns:
            numpy.ndarray, array of val nodes

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> val_nodes = dataset.val_nodes
        """
        return np.nonzero(self.val_mask)[0].astype(np.int32)

    @property
    def test_nodes(self):
        """
        Test nodes indexes

        Returns:
            numpy.ndarray, array of test nodes

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> test_nodes = dataset.test_nodes
        """
        return np.nonzero(self.test_mask)[0].astype(np.int32)

    @property
    def node_count(self):
        """
        Number of nodes

        Returns:
            int, length of csr row

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
            int, length of csr col

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
            numpy.ndarray, array of node feature

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
            numpy.ndarray, array of node label

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> node_label = dataset.node_label
        """
        if self._node_label is None:
            self._node_label = self._npz_file["label"]
        return self._node_label.astype(np.int32)

    def __getitem__(self, idx):
        assert idx == 0, "reddit only has one graph"
        graph = MindHomoGraph()
        node_dict = {idx: idx for idx in range(self.node_count)}
        edge_ids = np.array(list(range(self.edge_count))).astype(np.int32)
        graph.set_topo(CsrAdj(self._csr_row, self._csr_col), node_dict=node_dict, edge_ids=edge_ids)
        return graph
