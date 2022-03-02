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
import os.path as osp
import numpy as np
from scipy.sparse import csr_matrix
from mindspore_gl.graph.graph import MindHomoGraph, CsrAdj


class BlogCatalog:
    """
    BlogCatalog Dataset, a source dataset for reading and parsing BlogCatalog dataset.

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

        self.load()

    def load(self):
        self._npz_file = np.load(self._path)
        self._csr_row = self._npz_file['adj_csr_indptr'].astype(np.int32)
        self._csr_col = self._npz_file['adj_csr_indices'].astype(np.int32)
        self._nodes = np.array(list(range(len(self._csr_row) - 1)))

    @property
    def num_classes(self):
        return 39

    @property
    def node_count(self):
        return len(self._csr_row)

    @property
    def edge_count(self):
        return len(self._csr_col)

    @property
    def node_label(self):
        if self._node_label is None:
            self._node_label = self._npz_file["label"]
        return self._node_label.astype(np.int32)

    @property
    def vocab(self):
        if self._vocab is None:
            self._vocab = self._npz_file["vocab"]
        return self._vocab.astype(np.int32)

    @property
    def adj_coo(self):
        return csr_matrix((np.ones(self._csr_col.shape), self._csr_col, self._csr_row)).tocoo(copy=False)

    @property
    def adj_csr(self):
        return csr_matrix((np.ones(self._csr_col.shape), self._csr_col, self._csr_row))

    def __getitem__(self, idx):
        assert idx == 0, "Blog Catalog only has one graph"
        graph = MindHomoGraph()
        node_dict = {idx: idx for idx in range(self.node_count)}
        edge_ids = np.array(list(range(self.edge_count))).astype(np.int32)
        graph.set_topo(CsrAdj(self._csr_row, self._csr_col), node_dict=node_dict, edge_ids=edge_ids)
        return graph
