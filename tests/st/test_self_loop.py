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
"""test self_loop"""
import scipy.sparse as sp
import mindspore as ms
from mindspore import Tensor, COOTensor
from mindspore import ops
from mindspore_gl.graph import remove_self_loop, add_self_loop


def test_remove_loop():
    """
    Feature:Test that the removal of self-loops is performed correctly.

    Description:Init a Graph
    src_idx = [0, 1, 2, 3]
    dst_idx = [0, 1, 2, 3]
    data = [1, 1, 1, 1]

    Expectation:
    src_idx = [0, 1, 2, 3]
    dst_idx = [0, 1, 2, 3]
    data = [0, 0, 0, 0]
    """

    adj = sp.csr_matrix(([1, 1, 1, 1], ([0, 1, 2, 3], [0, 1, 2, 3])), shape=(4, 4)).tocoo()
    adj_new = remove_self_loop(adj, mode='dense')
    for i in range(3):
        assert adj_new[i][i] == 0

    adj = sp.csr_matrix(([1, 2, 3, 4], ([0, 1, 2, 2], [0, 1, 2, 1])), shape=(3, 3)).tocoo()
    adj = remove_self_loop(adj, 'coo')
    assert ~adj.diagonal().any()

def test_add_loop():
    """
    Feature:Test that adding a self-loop is executed correctly.

    Description:Init a Graph
    src_idx = [0, 1, 2]
    dst_idx = [1, 2, 0]

    Expectation:
    src_idx = [0, 1, 2, 0, 1, 2]
    dst_idx = [1, 2, 0, 0, 1, 2]
    """

    indices = Tensor([[0, 1], [1, 2], [2, 0]], dtype=ms.int32)
    values = Tensor([1, 2, 1], dtype=ms.float32)
    node = 3
    shape = (node, node)
    adj = COOTensor(indices, values, shape)
    fill_value = Tensor([1, 1, 1], ms.float32)
    new_adj = add_self_loop(adj, node, fill_value, mode='dense')
    for i in range(node):
        assert new_adj[i][i] != 0

    new_adj = add_self_loop(adj, node, fill_value, mode='coo')
    edge_index = new_adj.indices
    edge_index = ops.Transpose()(edge_index, (1, 0))
    count = 0
    for i in range(edge_index.shape[1]):
        if edge_index[0, i] == edge_index[1, i]:
            count += 1
    assert count >= node
