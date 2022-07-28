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
"""Test Conv Layers."""
import pytest
import numpy as np
import mindspore as ms
from mindspore_gl import GraphField, BatchedGraphField
from mindspore_gl.nn.conv import GCNConv2

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
def test_gcnconv2():
    """
    Features: GCNConv2
    Description: Test GCNConv2.
    Expectation: The output is as expected.
    """
    weight_gcn = np.array([-0.21176505, -0.33282989, 0.32413161, 0.44888139])
    weight_gcn = ms.Tensor(weight_gcn, ms.float32).view((1, 4))
    bias_gcn = np.array(0.34598231)
    bias_gcn = ms.Tensor(bias_gcn, ms.float32).view(1)
    weight_gcn2 = np.array([-0.48440778, -0.30549908, 0.08788288, 0.25465935])
    weight_gcn2 = ms.Tensor(weight_gcn2, ms.float32).view((1, 4))
    expect_output = np.array([[1.76639652], [2.13106108], [0.13948047], [-3.51022506],
                              [-3.67287588], [-2.50677252], [-0.1081664]])

    net = GCNConv2(4, 1)
    net.fc1.weight.set_data(weight_gcn)
    net.fc2.weight.set_data(weight_gcn2)
    net.bias.set_data(bias_gcn)
    output = net(node_feat, *graph_field.get_graph())
    assert np.allclose(output.asnumpy(), expect_output)
