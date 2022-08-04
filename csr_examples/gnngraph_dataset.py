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
"""Full training dataset"""
import numpy as np
import mindspore as ms
import scipy.sparse


class GraphDataset:
    """Full training numpy dataset """
    def __init__(self, data_path: str) -> None:
        npz = np.load(data_path)
        self.n_nodes = npz.get('n_nodes', default=npz['feat'].shape[0])
        self.n_edges = npz.get('n_edges', default=npz['adj_coo_row'].shape[0])
        self.n_classes = int(npz.get('n_classes', default=np.max(npz['label']) + 1))
        row_indices = np.asarray(npz["adj_coo_row"], dtype=np.int32)
        col_indices = np.asarray(npz["adj_coo_col"], dtype=np.int32)
        out_deg = np.bincount(npz['adj_coo_row'], minlength=self.n_nodes)
        in_deg = np.bincount(npz['adj_coo_col'], minlength=self.n_nodes)
        in_deg = npz.get('in_degrees', default=in_deg)
        out_deg = npz.get('out_degrees', default=out_deg)

        side = int(self.n_nodes)
        nnz = row_indices.shape[0]
        idx_forward = np.argsort(out_deg)[::-1]
        arg_idx_forward = np.argsort(idx_forward)

        row_indices_forward = arg_idx_forward[row_indices]
        col_indices_forward = arg_idx_forward[col_indices]
        coo_tensor_forward = scipy.sparse.coo_matrix(
            (np.ones(nnz), (row_indices_forward, col_indices_forward)), shape=(side, side))
        csr_tensor_forward = coo_tensor_forward.tocsr()

        self.row_indices = ms.Tensor(np.sort(row_indices_forward), dtype=ms.int32)
        self.indptr = ms.Tensor(np.asarray(csr_tensor_forward.indptr), dtype=ms.int32)
        self.indices = ms.Tensor(np.asarray(csr_tensor_forward.indices), dtype=ms.int32)

        coo_tensor_backward = scipy.sparse.csr_matrix(
            (np.ones(nnz), (col_indices_forward, row_indices_forward)), shape=(side, side))
        csr_tensor_backward = coo_tensor_backward.tocsr()

        self.indptr_backward = ms.Tensor(np.asarray(csr_tensor_backward.indptr), dtype=ms.int32)
        self.indices_backward = ms.Tensor(np.asarray(csr_tensor_backward.indices), dtype=ms.int32)

        self.x = ms.Tensor(npz['feat'][idx_forward])
        self.y = ms.Tensor(npz['label'][idx_forward], ms.int32)
        self.train_mask = npz.get('train_mask', default=None)
        if self.train_mask is not None:
            self.train_mask = self.train_mask[idx_forward]
        self.test_mask = npz.get('test_mask', default=None)
        if self.test_mask is not None:
            self.test_mask = self.test_mask[idx_forward]
        self.in_deg = ms.Tensor(in_deg[idx_forward], ms.int32)
        self.out_deg = ms.Tensor(out_deg[idx_forward], ms.int32)
