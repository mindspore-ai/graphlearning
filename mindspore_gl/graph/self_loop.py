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
""" self_loop """
import numpy as np
import scipy.sparse as sp
import mindspore as ms
from mindspore import ops
import mindspore.numpy as mp
import mindspore.nn as nn
from mindspore import COOTensor

def remove_self_loop(adj, mode='dense'):
    """
    Remove the diagonal matrix from the input matrix object,
    you can choose to operate on a dense matrix or a matrix in coo format.

    Args:
        adj(scipy.sparse.coo): Target matrix.
        mode(str): type of operation matrix. Default: dense.

    Returns:
        - **adj** (scipy.sparse.coo) - The object after removing the diagonal matrix.
          'dense' returns the Tensor type.
          'coo' returns the scipy.sparse.coo type.

    Examples:
        >>> from mindspore_gl.graph.self_loop import remove_self_loop
        >>> import scipy.sparse as sp
        >>> adj = sp.csr_matrix(([1, 2, 3, 4], ([0, 1, 2, 2], [0, 1, 2, 1])), shape=(3, 3)).tocoo()
        >>> adj = remove_self_loop(adj, 'coo')
        >>> print(adj)
            (1, 2)        4
    """
    if mode == 'dense':
        shape = adj.toarray().shape
        mask = np.ones(shape)
        mask[:shape[0]].flat[::shape[0]+1] = False
        adj_new = adj.toarray() * mask
        adj = ms.Tensor(adj_new, ms.float32)
    elif mode == 'coo':
        mask = adj.col != adj.row
        adj = sp.csr_matrix((adj.data[mask], (adj.col[mask], adj.row[mask])), shape=adj.shape).tocoo()
    else:
        raise ValueError('Other formats are not currently supported.')

    return adj

def add_self_loop(edge_index, edge_weight, node, fill_value, mode='dense'):
    r"""
    ADD the self loop from the input coo matrix.
    you can choose to operate on a dense matrix or a matrix in coo format.

    Args:
        edge_index (Tensor): Edge index. The shape is :math:`(2, N\_e)`
            where :math:`N\_e` is the number of edges.
        edge_weight (Tensor): Edge weights. The shape is :math:`(N\_e)`
            where :math:`N\_e` is the number of edges.
        node(int): Number of nodes.
        fill_value(Tensor): self-loop value.
        mode(str): type of operation matrix. Default: dense.

    Returns:
        if `mode` is 'dense',

        - **new_adj** (Tensor) - dense matrix.

        if `mode` is 'coo',

        - **edge_index** (Tensor) - new edge_index.
        - **edge_weight** (Tensor) - new edge_weight

    Raises:
        ValueError: if `mode` not is coo or dense.
        ValueError: if `fill_value` length not equal to `node`.
        TypeError: If `node` is not a positive int.

    Examples:
        >>> from mindspore import Tensor
        >>> from mindspore_gl.graph import add_self_loop
        >>> edge_index = [[1, 1, 2, 2], [0, 2, 0, 1]]
        >>> edge_index = ms.Tensor(edge_index, ms.int32)
        >>> edge_weight = Tensor([1, 1, 1, 1], ms.float32)
        >>> node = 3
        >>> fill_value = Tensor([2, 2, 2], ms.float32)
        >>> new_adj = add_self_loop(edge_index, edge_weight, node, fill_value, mode='dense')
        >>> print(new_adj)
        [[2. 0. 0.]
         [1. 2. 1.]
         [1. 1. 2.]]
        >>> edge_index, edge_weight = add_self_loop(edge_index, edge_weight, node, fill_value, mode='coo')
        >>> print(edge_index)
        [[1 1 2 2 0 1 2]
         [0 2 0 1 0 1 2]]
        >>> print(edge_weight)
        [1. 1. 1. 1. 2. 2. 2.]
    """
    if not isinstance(node, int):
        raise TypeError("The node data type is {},\
                        but it should be int.".format(type(node)))
    if mode not in ['coo', 'dense']:
        raise TypeError("The node type is {},\
                                but it should be 'coo' or 'dense'.".format(type(mode)))
    if fill_value.shape[0] != node:
        raise ValueError("The fill_value length must equal to node")
    indices = ops.Transpose()(edge_index, (1, 0))
    shape = (node, node)
    adj = ms.COOTensor(indices, edge_weight, shape)
    shape = adj.shape
    range_index = nn.Range(0, node, 1)
    loop_index = range_index()
    loop_index = ops.ExpandDims()(loop_index, 0)
    loop_index = mp.tile(loop_index, (2, 1))
    loop_index = ops.Transpose()(loop_index, (1, 0))
    edge_index = adj.indices
    edge_index = ops.Concat()((edge_index, loop_index))
    edge_attr = adj.values
    edge_attr = ops.Concat()((edge_attr, fill_value))
    new_adj = COOTensor(edge_index, edge_attr, shape)
    if mode == 'dense':
        new_adj = new_adj.to_dense()
        return new_adj
    edge_index = new_adj.indices
    edge_index = ops.Transpose()(edge_index, (1, 0))
    edge_weight = new_adj.values
    return edge_index, edge_weight
