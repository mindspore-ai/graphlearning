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
"""datasets"""
import os
from six.moves import urllib
import numpy as np
from scipy.sparse import coo_matrix
import mindspore as ms
from mindspore_gl.graph import MindHomoGraph, CsrAdj
from mindspore_gl.sampling import random_walk_unbias_on_homo


class Texas:
    r"""
    Texas Dataset, a source dataset for reading and parsing Cora dataset.

    Args:
        root(str): path to the root directory that contains raw data.
        name(str): select dataset type, optional: ["texas"].
        pre_transform: Pretransform node features

    Dataset can be download here:
    <https://github.com/graphdml-uiuc-jlu/geom-gcn/tree/master/new_data/texas>
    """
    def __init__(self, root, name='texas', pre_transform=True):
        if not isinstance(root, str):
            raise TypeError(
                f"For '{self.cls_name}', the 'root' should be a str, "
                f"but got {type(root)}.")

        self._root = root
        self._name = name
        self._path = os.path.join(root, self._name)

        self.x = None
        self.y = None
        self.edge_index = None
        self.pre_transform = pre_transform

        self._csr_row = None
        self._csr_col = None
        self._nodes = None
        self._indegree = None
        self._outdegree = None

        self._num_classes = None
        self.load()
        self.build_degree()

        self.isolated_nodes = \
            np.nonzero((self._indegree == 0) & (self._outdegree == 0))[0]
        self.num_isolated_nodes = len(self.isolated_nodes)
        if self.num_isolated_nodes > 0:
            self._indegree[self.isolated_nodes] = 1
            self._outdegree[self.isolated_nodes] = 1
            self.edge_index = np.concatenate([self.edge_index, np.stack(
                [self.isolated_nodes, self.isolated_nodes])])

        self.build_csr()

    def downloads(self):
        """ Download data"""
        os.makedirs(self._path, exist_ok=True)
        feature_file = os.path.join(self._path, 'out1_node_feature_label.txt')
        graph_file = os.path.join(self._path, 'out1_graph_edges.txt')
        if not os.path.exists(feature_file):
            print('download file out1_node_feature_label.txt !!!')
            origin = 'https://raw.githubusercontent.com/graphdml-uiuc-jlu/' + \
                    'geom-gcn/master/new_data/texas/out1_node_feature_label.txt'
            urllib.request.urlretrieve(origin, feature_file)
        if not os.path.exists(graph_file):
            print('download file out1_graph_edges.txt !!!')
            origin = 'https://raw.githubusercontent.com/graphdml-uiuc-jlu/' + \
                     'geom-gcn/master/new_data/texas/out1_graph_edges.txt'
            urllib.request.urlretrieve(origin, graph_file)

    def load(self):
        """Load and process data"""
        self.downloads()
        with open(os.path.join(self._path, 'out1_node_feature_label.txt'),
                  'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            self.x = np.asarray(x, dtype=np.float)

            y = [int(r.split('\t')[2]) for r in data]
            self.y = np.asarray(y, dtype=np.long)

        with open(os.path.join(self._path, 'out1_graph_edges.txt'), 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [tuple(int(v) for v in r.split('\t')) for r in data]
            data = data + [(x[1], x[0]) for x in data]
            data = set(data)
            data = [list(x) for x in data]
            self.edge_index = np.ascontiguousarray(
                np.array(data, dtype=np.long).transpose())

        if self.pre_transform:
            # currently using a Normalized transform to x
            self.x = self.x - self.x.min()
            self.x = self.x / np.clip(self.x.sum(axis=-1, keepdims=True),
                                      a_min=1.0, a_max=None)

    @staticmethod
    def to_undirected(edge_index):
        row, col = edge_index
        row, col = np.concatenate([row, col], axis=0), np.concatenate(
            [col, row], axis=0)
        edge_index = np.stack([row, col], axis=0)
        return edge_index

    def build_csr(self):
        if self._csr_col is None:
            tmp_a = coo_matrix((np.ones_like(self.edge_index[0]), (
                self.edge_index[0], self.edge_index[1]))).tocsr()
            self._csr_row = tmp_a.indptr
            self._csr_col = tmp_a.indices
            self._nodes = np.array(list(range(len(self._csr_row) - 1)))

    def build_degree(self):
        if self._indegree is None:
            self._indegree = np.zeros(shape=(self.num_nodes))
            for i in self.edge_index[1]:
                self._indegree[i] += 1
        if self._outdegree is None:
            self._outdegree = np.zeros(shape=(self.num_nodes))
            for i in self.edge_index[0]:
                self._outdegree[i] += 1

    @property
    def node_feat_size(self):
        return self.x.shape[-1]

    @property
    def num_classes(self):
        if self._num_classes is None:
            self._num_classes = len(np.unique(self.y))
        return self._num_classes

    @property
    def num_nodes(self):
        return self.x.shape[0]

    @property
    def num_edges(self):
        return self.edge_index.shape[-1]

    @property
    def indegree(self):
        return self._indegree

    @property
    def outdegree(self):
        return self._outdegree

    def __getitem__(self, idx):
        graph = MindHomoGraph()
        node_dict = {idx: idx for idx in range(self.num_nodes)}
        edge_ids = np.array(list(range(self.num_edges))).astype(np.int32)
        graph.set_topo(CsrAdj(self._csr_row, self._csr_col),
                       node_dict=node_dict, edge_ids=edge_ids)
        return graph


def multilyer_rwsampling(graph, args, nlayer):
    """Multi-layer Heterophily-Sampling"""
    starts = list()
    ends = list()
    end_indexs = list()
    for _ in range(nlayer):
        st, en, eni = rwsampling(graph, args)
        starts.append(st)
        ends.append(en)
        end_indexs.append(eni)
    starts = ms.Tensor(starts)
    ends = ms.Tensor(ends)
    end_indexs = ms.Tensor(end_indexs)
    return starts, ends, end_indexs


def rwsampling(graph, args):
    """one-layer Heterophily-Sampling"""
    batch_num = graph.node_count - 1
    k = args.k
    rws = args.rws
    batch = np.arange(0, batch_num, dtype=np.int32)
    rws_1 = rws * (k + 1)
    rws_n = batch_num * rws_1

    overall = np.arange(0, rws_n)
    start_index = overall // rws_1
    starts = batch[start_index]
    end_index = overall % (k + 1)
    ends = random_walk_unbias_on_homo(graph, starts, k)
    ends = ends[overall, end_index]

    return starts, ends, end_index
