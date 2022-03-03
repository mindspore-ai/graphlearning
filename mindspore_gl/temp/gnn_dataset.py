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

from mindspore_gl import GraphField


class GraphDataset:
    """Full training numpy dataset """
    def __init__(self, data_path: str) -> None:
        npz = np.load(data_path)
        self.x = ms.Tensor(npz['feat'])
        self.y = ms.Tensor(npz['label'], ms.int32)
        self.train_mask = npz.get('train_mask', default=None)
        self.test_mask = npz.get('test_mask', default=None)
        self.n_nodes = npz.get('n_nodes', default=self.x.shape[0])
        self.n_edges = npz.get('n_edges', default=npz['adj_coo_row'].shape[0])
        self.g = GraphField(ms.Tensor(npz['adj_coo_row'], dtype=ms.int32),
                            ms.Tensor(npz['adj_coo_col'], dtype=ms.int32),
                            int(self.n_nodes),
                            int(self.n_edges))
        self.n_classes = int(npz.get('n_classes', default=np.max(npz['label']) + 1))
        in_deg = np.zeros(shape=self.n_nodes, dtype=np.int)
        out_deg = np.zeros(shape=self.n_nodes, dtype=np.int)
        for r in npz['adj_coo_row']:
            out_deg[r] += 1
        for c in npz['adj_coo_col']:
            in_deg[c] += 1
        self.in_deg = ms.Tensor(npz.get('in_degrees', default=in_deg), ms.int32)
        self.out_deg = ms.Tensor(npz.get('out_degrees', default=out_deg), ms.int32)
