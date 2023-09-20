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
"""Convert the coo graph to the csr graph."""
import numpy as np
import mindspore as ms
import scipy


def csr_data(row_indices, col_indices, n_nodes, n_edges):
    """Convert the COO format to the CSR format."""
    coo_tensor_forward = scipy.sparse.coo_matrix(
        (np.ones(n_edges), (row_indices, col_indices)), shape=(n_nodes, n_nodes))
    csr_tensor_forward = coo_tensor_forward.tocsr()
    indptr = np.asarray(csr_tensor_forward.indptr, np.int32)
    indices = np.asarray(csr_tensor_forward.indices, np.int32)
    coo_tensor_backward = scipy.sparse.csr_matrix(
        (np.ones(n_edges), (col_indices, row_indices)), shape=(n_nodes, n_nodes))
    csr_tensor_backward = coo_tensor_backward.tocsr()
    indptr_backward = ms.Tensor(np.asarray(csr_tensor_backward.indptr), dtype=ms.int32)
    indices_backward = ms.Tensor(np.asarray(csr_tensor_backward.indices), dtype=ms.int32)
    indices = ms.Tensor(indices, ms.int32)
    indptr = ms.Tensor(indptr, ms.int32)
    return indices, indptr, indices_backward, indptr_backward

def rerank_index(out_deg, row_indices, col_indices):
    """reorder the index according to the out degree"""
    idx_forward = np.argsort(out_deg)[::-1]
    arg_idx_forward = np.argsort(idx_forward)
    arg_idx_forward = np.array(arg_idx_forward, np.int32)
    row_indices_forward = arg_idx_forward[row_indices]
    col_indices_forward = arg_idx_forward[col_indices]
    return row_indices_forward, col_indices_forward, idx_forward, arg_idx_forward

def graph_csr_data(src_idx, dst_idx, n_nodes, n_edges, node_feat=None, node_label=None, train_mask=None, val_mask=None,
                   test_mask=None, rerank=False):
    r"""
    Convert the entire graph in the COO format to the CSR format.

    Args:
        src_idx (Union[Tensor, numpy.ndarray]): tensor with shape :math:`(N\_EDGES)`, with int dtype,
            represents the source node index of COO edge matrix.
        dst_idx (Union[Tensor, numpy.ndarray]): tensor with shape :math:`(N\_EDGES)`, with int dtype,
            represents the destination node index of COO edge matrix.
        n_nodes (int): integer, represent the nodes count of the graph.
        n_edges (int): integer, represent the edges count of the graph.
        node_feat (Union[Tensor, numpy.ndarray, optional]): node feature. Default: ``None``.
        node_label (Union[Tensor, numpy.ndarray, optional]): node labels. Default: ``None``.
        train_mask (Union[Tensor, numpy.ndarray, optional]): mask of train index. Default: ``None``.
        val_mask (Union[Tensor, numpy.ndarray, optional]): msk of train index. Default: ``None``.
        test_mask (Union[Tensor, numpy.ndarray, optional]): mask of train index. Default: ``None``.
        rerank (bool, optional): whether to reorder node features, node labels, and masks.
            Default: ``False``.

    Returns:
        - **csr_g** (tuple) - info of csr graph, it contains indices of csr graph, indptr of csr graph,
            node numbers of csr graph, edges numbers of csr graph, pre-stored backward indices of csr graph,
            pre-stored backward indptr of csr graph.
        - **in_deg** - in degree of each node.
        - **out_deg** - out degree of each node.
        - **node_feat** (Union[Tensor, numpy.ndarray, optional]) - reorder node features.
        - **node_label** (Union[Tensor, numpy.ndarray, optional]) - reorder node labels.
        - **train_mask** (Union[Tensor, numpy.ndarray, optional]) - reorder train index mask.
        - **val_mask** (Union[Tensor, numpy.ndarray, optional]) - reorder val index mask.
        - **test_mask** (Union[Tensor, numpy.ndarray, optional]) - reorder test index mask.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore_gl.graph import graph_csr_data
        >>> node_feat = np.array([[1, 2, 3, 4], [2, 4, 1, 3], [1, 3, 2, 4],
        ...                       [9, 7, 5, 8], [8, 7, 6, 5], [8, 6, 4, 6], [1, 2, 1, 1]], np.float32)
        >>> n_nodes = 7
        >>> n_edges = 8
        >>> edge_feat_size = 7
        >>> src_idx = np.array([0, 2, 2, 3, 4, 5, 5, 6], np.int32)
        >>> dst_idx = np.array([1, 0, 1, 5, 3, 4, 6, 4], np.int32)
        >>> node_label = np.array([0, 1, 0, 1, 0, 1, 0])
        >>> train_mask = np.array([True, True, True, True, False, False, False])
        >>> val_mask = np.array([False, False, False, False, True, True, True])
        >>> g, in_deg, out_deg, node_feat, node_label, train_mask, val_mask,\
        >>> test_mask = graph_csr_data(src_idx,dst_idx, n_nodes, n_edges, node_feat, node_label,
        ...                            train_mask, val_mask, test_mask=None, rerank=True)
        >>> print(g[0], g[1])
        [2 3 5 6 3 4 0 6] [0 2 4 5 6 7 8 8]
        >>> print(node_feat, node_label)
        [[8. 7. 6. 5.]
        [2. 4. 1. 3.]
        [1. 2. 1. 1.]
        [8. 6. 4. 6.]
        [9. 7. 5. 8.]
        [1. 2. 3. 4.]
        [1. 3. 2. 4.]] [0 1 0 1 1 0 0]
        >>> print(train_mask, val_mask)
        [False  True False False  True  True  True] [ True False  True  True False False False]
    """
    if isinstance(dst_idx, ms.Tensor):
        dst_idx = dst_idx.asnumpy()
    if isinstance(src_idx, ms.Tensor):
        src_idx = src_idx.asnumpy()
    row_indices = np.array(dst_idx, np.int32)
    col_indices = np.array(src_idx, np.int32)
    out_deg = np.bincount(row_indices, minlength=n_nodes)
    in_deg = np.bincount(col_indices, minlength=n_nodes)
    if not rerank:
        indices, indptr, indices_backward, indptr_backward = csr_data(row_indices, col_indices, n_nodes, n_edges)
    else:
        row_indices_forward, col_indices_forward, idx_forward, _ = rerank_index(out_deg, row_indices, col_indices)
        indices, indptr, indices_backward, indptr_backward = csr_data(row_indices_forward, col_indices_forward,
                                                                      n_nodes, n_edges)
        idx_forward = idx_forward.tolist()
        in_deg = in_deg[idx_forward]
        out_deg = out_deg[idx_forward]
        node_feat = node_feat[idx_forward]
        node_label = node_label[idx_forward]
        if train_mask is not None:
            train_mask = train_mask[idx_forward]
        if val_mask is not None:
            val_mask = val_mask[idx_forward]
        if test_mask is not None:
            test_mask = test_mask[idx_forward]
    in_deg = ms.Tensor(in_deg, ms.int32)
    out_deg = ms.Tensor(out_deg, ms.int32)
    csr_g = (indices, indptr, n_nodes, n_edges, indices_backward, indptr_backward)
    return csr_g, in_deg, out_deg, node_feat, node_label, train_mask, val_mask, test_mask

def sampling_csr_data(src_idx, dst_idx, n_nodes, n_edges, seeds_idx=None, node_feat=None, rerank=False):
    r"""
    Convert the sampling graph in the COO format to the CSR format.

    Args:
        src_idx (Union[Tensor, numpy.ndarray]): tensor with shape :math:`(N\_EDGES)`, with int dtype,
            represents the source node index of COO edge matrix.
        dst_idx (Union[Tensor, numpy.ndarray]): tensor with shape :math:`(N\_EDGES)`, with int dtype,
            represents the destination node index of COO edge matrix.
        n_nodes (int): integer, represent the nodes count of the graph.
        n_edges (int): integer, represent the edges count of the graph.
        seeds_idx (numpy.ndarray): start nodes for neighbor sampling. Default: ``None``.
        node_feat (Union[Tensor, numpy.ndarray], optional): node feature. Default: ``None``.
        rerank (bool, optional): whether to reorder node features, node labels, and masks.
            Default: ``False``.

    Returns:
        - **csr_g** (tuple) - info of csr graph, it contains indices of csr graph, indptr of csr graph,
          node numbers of csr graph, edges numbers of csr graph, pre-stored backward indices of csr graph,
          pre-stored backward indptr of csr graph.
        - **seeds_idx** (numpy.ndarray) - reordered start nodes.
        - **node_feat** (numpy.ndarray) - reorder node features.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore_gl.graph import sampling_csr_data
        >>> node_feat = np.array([[1, 2, 3, 4], [2, 4, 1, 3], [1, 3, 2, 4],
        ...                       [9, 7, 5, 8], [8, 7, 6, 5], [8, 6, 4, 6], [1, 2, 1, 1]], np.float32)
        >>> n_nodes = 7
        >>> n_edges = 8
        >>> edge_feat_size = 7
        >>> src_idx = np.array([0, 2, 2, 3, 4, 5, 5, 6], np.int32)
        >>> dst_idx = np.array([1, 0, 1, 5, 3, 4, 6, 4], np.int32)
        >>> seeds_idx = np.array([0, 3, 5])
        >>> g, seeds_idx, node_feat = sampling_csr_data(src_idx, dst_idx, n_nodes, n_edges,\
        ...                                             seeds_idx, node_feat, rerank=True)
        >>> print(g[0], g[1], seeds_idx)
        [2 3 5 6 3 4 0 6] [0 2 4 5 6 7 8 8] [5, 4, 3]
        >>> print(node_feat)
        [[8. 7. 6. 5.]
         [2. 4. 1. 3.]
         [1. 2. 1. 1.]
         [8. 6. 4. 6.]
         [9. 7. 5. 8.]
         [1. 2. 3. 4.]
         [1. 3. 2. 4.]]
    """
    if isinstance(dst_idx, ms.Tensor):
        dst_idx = dst_idx.asnumpy()
    if isinstance(src_idx, ms.Tensor):
        src_idx = src_idx.asnumpy()
    row_indices = np.array(dst_idx, np.int32)
    col_indices = np.array(src_idx, np.int32)
    out_deg = np.bincount(row_indices, minlength=n_nodes)
    if not rerank:
        indices, indptr, indices_backward, indptr_backward = csr_data(row_indices, col_indices, n_nodes, n_edges)
    else:
        row_indices_forward, col_indices_forward, idx_forward, arg_idx_forward = rerank_index(out_deg, row_indices,
                                                                                              col_indices)
        indices, indptr, indices_backward, indptr_backward = csr_data(row_indices_forward, col_indices_forward,
                                                                      n_nodes, n_edges)
        idx_forward = idx_forward.tolist()
        origin_idx = list(range(len(arg_idx_forward)))
        idx_dict = dict(zip(origin_idx, arg_idx_forward))
        seeds_idx = [idx_dict[i] for i in seeds_idx]
        node_feat = node_feat[idx_forward, :]
    csr_g = (indices, indptr, n_nodes, n_edges, indices_backward, indptr_backward)
    return csr_g, seeds_idx, node_feat

def batch_graph_csr_data(src_idx, dst_idx, n_nodes, n_edges, node_map_idx, node_feat=None, rerank=False):
    r"""
    Convert the batched graph in the COO format to the CSR format.

    Args:
        src_idx (Union[Tensor, numpy.ndarray]): tensor with shape :math:`(N\_EDGES)`, with int dtype,
            represents the source node index of COO edge matrix.
        dst_idx (Union[Tensor, numpy.ndarray]): tensor with shape :math:`(N\_EDGES)`, with int dtype,
            represents the destination node index of COO edge matrix.
        n_nodes (int): integer, represent the nodes count of the graph.
        n_edges (int): integer, represent the edges count of the graph.
        node_map_idx (numpy.ndarray): ID of the subgraph to each node belongs to.
        node_feat (Union[Tensor, numpy.ndarray, optional]): node feature. Default: ``None``.
        rerank (bool, optional): whether to reorder node features, node labels, and masks.
            Default: ``False``.

    Returns:
        - **csr_g** (tuple) - info of csr graph, it contains indices of csr graph, indptr of csr graph,
          node numbers of csr graph, edges numbers of csr graph, pre-stored backward indices of csr graph,
          pre-stored backward indptr of csr graph.
        - **node_map_idx** (numpy.ndarray) - reordered start map index.
        - **node_feat** (Union[Tensor, numpy.ndarray, optional]) - reorder node features.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore_gl.graph import batch_graph_csr_data
        >>> node_feat = np.array([[1, 2, 3, 4], [2, 4, 1, 3], [1, 3, 2, 4],
        ...                       [9, 7, 5, 8], [8, 7, 6, 5], [8, 6, 4, 6], [1, 2, 1, 1]], np.float32)
        >>> n_nodes = 7
        >>> n_edges = 8
        >>> edge_feat_size = 7
        >>> src_idx = np.array([0, 2, 2, 3, 4, 5, 5, 6], np.int32)
        >>> dst_idx = np.array([1, 0, 1, 5, 3, 4, 6, 4], np.int32)
        >>> node_map_idx = np.array([0, 0, 0, 0, 1, 1, 1])
        >>> g, node_map_idx, node_feat = batch_graph_csr_data(src_idx, dst_idx,\
        ...                                                   n_nodes, n_edges, node_map_idx, node_feat, rerank=True)
        >>> print(g[0], g[1], node_map_idx)
        [2 3 5 6 3 4 0 6] [0 2 4 5 6 7 8 8] [1 0 1 1 0 0 0]
        >>> print(node_feat)
        [[8. 7. 6. 5.]
         [2. 4. 1. 3.]
         [1. 2. 1. 1.]
         [8. 6. 4. 6.]
         [9. 7. 5. 8.]
         [1. 2. 3. 4.]
         [1. 3. 2. 4.]]
    """
    if isinstance(dst_idx, ms.Tensor):
        dst_idx = dst_idx.asnumpy()
    if isinstance(src_idx, ms.Tensor):
        src_idx = src_idx.asnumpy()
    row_indices = np.array(dst_idx, np.int32)
    col_indices = np.array(src_idx, np.int32)
    out_deg = np.bincount(row_indices, minlength=n_nodes)
    if not rerank:
        indices, indptr, indices_backward, indptr_backward = csr_data(row_indices, col_indices, n_nodes, n_edges)
    else:
        row_indices_forward, col_indices_forward, idx_forward, _ = rerank_index(out_deg, row_indices,
                                                                                col_indices)
        indices, indptr, indices_backward, indptr_backward = csr_data(row_indices_forward, col_indices_forward, n_nodes,
                                                                      n_edges)
        idx_forward = idx_forward.tolist()
        node_feat = node_feat[idx_forward]
        node_map_idx = node_map_idx[idx_forward]
    csr_g = (indices, indptr, n_nodes, n_edges, indices_backward, indptr_backward)
    return csr_g, node_map_idx, node_feat
