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

"""GCN normalization"""
import mindspore as ms
from mindspore import ops
from mindspore_gl.graph.self_loop import add_self_loop

def gcn_norm(edge_index, n_nodes):
    r"""
    Normalization for GCNEConv

    Args:
        edge_index (Tensor): Edge index. The shape is :math:`(2, N\_e)`
            where :math:`N\_e` is the number of edges.
        n_nodes (int): Number of nodes.

    Returns:
        - **edge_index** (Tensor) - normalized edge_index. The shape is :math:`(2, N\_normed\_e)`
        - **edge_weight** (Tensor) - normalized edge_weight. The shape is :math:`(N\_normed\_e, 1)`

    Raises:
        TypeError: if `n_nodes` is not a positive int.
        TypeError: if `edge_index` type is not the `mindspore.Tensor`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore_gl.graph import gcn_norm
        >>> edge_index = [[1, 1, 2, 2], [0, 2, 0, 1]]
        >>> edge_index = ms.Tensor(edge_index, ms.int32)
        >>> num_nodes = 3
        >>> edge_index, edge_weight = gcn_norm(edge_index, num_nodes)
        >>> print(edge_index)
        [[1 1 2 2 0 1 2]
         [0 2 0 1 0 1 2]]
        >>> print(edge_weight)
        [[0.40824825]
         [0.4999999 ]
         [0.40824825]
         [0.4999999 ]
         [0.3333333 ]
         [0.4999999 ]
         [0.4999999 ]]
    """
    if not isinstance(n_nodes, int) or n_nodes <= 0:
        raise TypeError("the 'n_nodes' must be a positive int")
    if not isinstance(edge_index, ms.Tensor):
        raise TypeError("the 'edge_index' data type must be mindspore.Tensor")
    src_idx = edge_index[0]
    dst_idx = edge_index[1]
    n_edges = edge_index.shape[1]
    min_clip, max_clip = 0, 1e6
    expend_dim = ops.ExpandDims()
    edge_weight = ops.ones((n_edges,), ms.float32)
    fill_value = ops.ones((n_nodes,), ms.float32)
    src_idx = expend_dim(src_idx, 0)
    dst_idx = expend_dim(dst_idx, 0)
    new_edge_index = ops.Concat()((src_idx, dst_idx))
    new_edge_index, edge_weight = add_self_loop(new_edge_index, edge_weight, n_nodes, fill_value, mode='coo')
    add_length = new_edge_index.shape[1]
    row, col = new_edge_index[0], new_edge_index[1]
    full_weight = ops.zeros((add_length,), ms.float32)
    indices = expend_dim(col, 1)
    op = ops.TensorScatterAdd()
    deg = op(full_weight, indices, edge_weight)[:n_nodes]
    deg_inv_sqrt = ops.Pow()(deg, -0.5)
    deg_inv_sqrt = ms.ops.clip_by_value(deg_inv_sqrt, min_clip, max_clip)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    edge_weight = expend_dim(edge_weight, 1)
    return new_edge_index, edge_weight
