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
import mindspore as ms
import numpy as np
import scipy.sparse as sp

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
    ADD the diagonal matrix from the input matrix object,
    you can choose to operate on a dense matrix or a matrix in coo format

    Args:
        adj(scipy.sparse.coo): Target matrix
        node(int): Number of nodes
        fill_value(array): self-loop value,shape:(node)
        mode(str): type of operation matrix

    Returns:
        The object after adding the diagonal matrix
        'dense' returns the Tensor type
        'coo' returns scipy.sparse.coo type

    Examples:
        >>> from mindspore_gl.graph.self_loop import add_self_loop
        >>> import scipy.sparse as sp
        >>> adj = sp.csr_matrix(([0, 0, 0], ([0, 1, 2], [0, 1, 2])), shape=(3, 3)).tocoo()
        >>> adj = add_self_loop(adj, 3, [1, 1, 1], 'coo')
        >>> print(adj)
          (0, 0)        1
          (1, 1)        1
          (2, 2)        1
    """
    if mode == 'dense':
        adj_new = adj.toarray() + np.diag(fill_value)
        adj = ms.Tensor(adj_new, ms.float32)
    elif mode == 'coo':
        loop = np.array([i for i in range(node)])
        data = np.concatenate([adj.data, fill_value])
        col = np.concatenate([adj.col, loop])
        row = np.concatenate([adj.row, loop])
        adj = sp.csr_matrix((data, (row, col)), shape=adj.shape).tocoo()
    else:
        raise ValueError('Other formats are not currently supported.')

    return adj
