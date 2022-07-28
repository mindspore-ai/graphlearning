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
"""Test Glob Layers."""
import pytest
import numpy as np
import mindspore as ms
from mindspore_gl import GraphField, BatchedGraphField
from mindspore_gl.nn.glob import SAGPooling

node_feat = ms.Tensor([
    [1, 2, 3, 4],
    [2, 4, 1, 3],
    [1, 3, 2, 4],
    [9, 7, 5, 8],
    [8, 7, 6, 5],
    [8, 6, 4, 6],
    [1, 2, 1, 1]
], ms.float32)

n_nodes = 7
n_edges = 8
src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
in_degree = np.zeros(shape=n_nodes, dtype=np.int)
out_degree = np.zeros(shape=n_nodes, dtype=np.int)
for r in src_idx:
    out_degree[r] += 1
for r in dst_idx:
    in_degree[r] += 1
in_degree = ms.Tensor(in_degree, ms.int32)
out_degree = ms.Tensor(out_degree, ms.int32)
ver_subgraph_idx = ms.Tensor([0, 0, 0, 0, 0, 0, 0], ms.int32)
edge_subgraph_idx = ms.Tensor([0, 0, 0, 0, 0, 0, 0, 0], ms.int32)
graph_mask = ms.Tensor([1], ms.int32)
batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges,
                                        ver_subgraph_idx, edge_subgraph_idx, graph_mask)
graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)


# test
@pytest.mark.lebel0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sagpooling():
    """
    Features: SAGPooling
    Description: Test Sagpooling.
    Expectation: The output is as expected.
    """
    weight_gcn = np.array([-0.43797874, -0.24720865, -0.32895839, 0.29777485])
    weight_gcn = ms.Tensor(weight_gcn, ms.float32).view((1, 4))
    bias_gcn = np.array(-0.23705053)
    bias_gcn = ms.Tensor(bias_gcn, ms.float32).view(1)
    weight_gcn2 = np.array([0.38274187, -0.24117965, -0.35172099, -0.12423509])
    weight_gcn2 = ms.Tensor(weight_gcn2, ms.float32).view((1, 4))
    expect_feature = np.array([[[-0.94450444, -2.83351326, -1.88900888, -3.77801776],
                                [-0.98751843, -1.97503686, -2.96255541, -3.95007372],
                                [-1.97504234, -3.95008469, -0.98752117, -2.96256351],
                                [-0.99995297, -1.99990594, -0.99995297, -0.99995297]]])
    expect_src = np.array([1., 0., 0.])
    expect_dst = np.array([2., 1., 2.])
    expect_perm = np.array([2., 0., 1., 6.])
    expect_score = np.array([[-0.94450444], [-0.98751843], [-0.98752117], [-0.99995297]])
    net = SAGPooling(4)
    net.gnn.fc1.weight.set_data(weight_gcn)
    net.gnn.fc2.weight.set_data(weight_gcn2)
    net.gnn.bias.set_data(bias_gcn)
    feature, src, dst, perm, perm_score = net(node_feat, None, 7, 4,
                                              *batched_graph_field.get_batched_graph())
    assert np.allclose(feature.asnumpy(), expect_feature)
    assert np.allclose(src.asnumpy(), expect_src)
    assert np.allclose(dst.asnumpy(), expect_dst)
    assert np.allclose(perm.asnumpy(), expect_perm)
    assert np.allclose(perm_score.asnumpy(), expect_score)
