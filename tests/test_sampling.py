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
"""Unit Test for sampling """
import unittest
import numpy as np
import networkx
from scipy.sparse import csr_matrix
from mindspore_gl.graph import MindHomoGraph, CsrAdj
from mindspore_gl.sampling import sage_sampler_on_homo, random_walk_unbias_on_homo

def generate_graph(node_count, edge_prob=0.1):
    """generate graph"""
    graph = networkx.generators.random_graphs.fast_gnp_random_graph(node_count, edge_prob)
    edge_array = np.transpose(np.array(list(graph.edges)))
    row = edge_array[0]
    col = edge_array[1]
    data = np.zeros(row.shape)
    csr_mat = csr_matrix((data, (row, col)), shape=(node_count, node_count))
    generated_graph = MindHomoGraph()
    node_dict = {idx: idx for idx in range(node_count)}
    edge_count = col.shape[0]
    edge_ids = np.array(list(range(edge_count))).astype(np.int32)
    generated_graph.set_topo(CsrAdj(csr_mat.indptr.astype(np.int32), \
            csr_mat.indices.astype(np.int32)), node_dict, edge_ids)
    return generated_graph


class TestSamplers(unittest.TestCase):
    """Test samplers"""
    @classmethod
    def setUpClass(cls) -> None:
        cls.node_count = 10000
        cls.edge_prob = 0.1
        cls.graph = generate_graph(cls.node_count, cls.edge_prob)

    def test_sage_sampling(self):
        nodes = np.arange(0, self.node_count)
        sage_sampler_on_homo(homo_graph=self.graph, seeds=nodes[:10].astype(np.int32), neighbor_nums=[2, 2])
    def test_random_walk(self):
        nodes = np.arange(0, self.node_count)
        random_walk_unbias_on_homo(homo_graph=self.graph, seeds=nodes[:30].astype(np.int32), walk_length=10)
