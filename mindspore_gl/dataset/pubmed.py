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
"""PubMed Dataset"""
#pylint: disable=W0702

from typing import Optional
import os.path as osp
import numpy as np
from mindspore_gl.graph import MindHomoGraph, CsrAdj


class PubMed:
    """
    PubMed Dataset, a source dataset for reading and parsing PubMed dataset.

    Args:
        root(str): path to the root directory that contains pubmed_with_mask.npz

    Raises:
        TypeError: if `root` is not a str.
        RuntimeError: if `root` does not contain data files.

    Examples:
        >>> from mindspore_gl.dataset import PubMed
        >>> root = "path/to/pubmed"
        >>> dataset = PubMed(root)

    """

    def __init__(self, root: Optional[str] = None):
        if not isinstance(root, str):
            raise TypeError(f"For '{self.cls_name}', the 'root' should be a str, "
                            f"but got {type(root)}.")
        self._root = root
        self._path = osp.join(root, 'pubmed_with_mask.npz')

        self._csr_row = None
        self._csr_col = None
        self._nodes = None

        self._node_feat = None
        self._node_label = None

        self._train_mask = None
        self._val_mask = None
        self._test_mask = None

        self.load()

    def load(self):
        self._npz_file = np.load(self._path)
        self._csr_row = self._npz_file['adj_csr_indptr'].astype(np.int32)
        self._csr_col = self._npz_file['adj_csr_indices'].astype(np.int32)

        self._nodes = np.array(list(range(len(self._csr_row) - 1)))

    @property
    def num_features(self):
        return self.node_feat.shape[1]

    @property
    def num_classes(self):
        return len(np.unique(self.node_label))

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
    def train_nodes(self):
        return np.nonzero(self.train_mask)[0].astype(np.int32)

    @property
    def val_nodes(self):
        return np.nonzero(self.val_mask)[0].astype(np.int32)

    @property
    def test_nodes(self):
        return np.nonzero(self.test_mask)[0].astype(np.int32)

    @property
    def node_count(self):
        return len(self._csr_row) - 1

    @property
    def edge_count(self):
        return len(self._csr_col)

    @property
    def node_feat(self):
        if self._node_feat is None:
            self._node_feat = self._npz_file["feat"]

        return self._node_feat

    @property
    def node_label(self):
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
