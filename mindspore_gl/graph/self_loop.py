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
    you can choose to operate on a dense matrix or a matrix in coo format

    Args:
        adj(scipy.sparse.coo): Target matrix
        mode(str): type of operation matrix

    Returns:
        The object after removing the diagonal matrix
        'dense' returns the Tensor type
        'coo' returns the scipy.sparse.coo type

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

def add_self_loop(adj, node, fill_value, mode='dense'):
    """
    Feature:
    ADD the selp loop from the input coo matrix,
    you can choose to operate on a dense matrix or a matrix in coo format

    Args:
        adj(Tensor): COO matrix
        node(int): Number of nodes
        fill_value(Tensor): self-loop value
        mode(str): type of operation matrix

    Returns:
        The object after adding the diagonal matrix
        'dense' returns the dense Tensor type
        'coo' returns the coo Tensor type

    Raises:
        ValueError: if `mode` not is coo or dense.
        TypeError: If `node` is not a positive int.

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2], [2, 0]], dtype=ms.int32)
        >>> values = Tensor([1, 2, 1], dtype=ms.float32)
        >>> shape = (3, 3)
        >>> adj = COOTensor(indices, values, shape)
        >>> node = 3
        >>> fill_value = Tensor([1, 1, 1], ms.float32)
        >>> new_adj = add_self_loop(adj, node, fill_value, mode='dense')
        >>> print(new_adj)
        [[1. 1. 0.]
         [0. 1. 2.]
         [1. 0. 1.]]
         >>> new_adj = add_self_loop(adj, node, fill_value, mode='coo')
         >>> print(new_adj.indices)
         [[0 1]
          [1 2]
          [2 0]
          [0 0]
          [1 1]
          [2 2]]
         >>> print(new_adj.values)
         [1. 2. 1. 1. 1. 1.]
    """
    if not isinstance(node, int):
        raise TypeError("The node data type is {},\
                        but it should be int.".format(type(node)))
    assert mode in ['coo', 'dense'], 'Invalid mode'
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
