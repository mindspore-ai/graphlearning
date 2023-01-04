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
""" knn_graph """
import faiss
import mindspore as ms
import scipy.sparse as sp
import numpy as np

def knn_graph(feat, k: int, dis: int = None, \
              loop: bool = False, gpu: bool = True, device: int = 0):
    r"""
    Computes graph edges to the nearest k points,
    and returns the reconstructed graph.

    Args:
      feat(numpy.ndarray): Node Feature Matrix, shape is :math:`(N, F)`.
      k(int): k neighbors.
      dis(int): limit distance. Default: None.
      loop(bool): Whether to keep self-loop. Default: False.
      gpu(bool): gpu acceleration. Default: True.
      device(int): device number. Default: 0.

    Returns:
        - **coo** - Rebuilt graph

    Examples:
        >>> import numpy as np
        >>> from mindspore_gl.sampling import knn_graph
        >>> f = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=np.float32)
        >>> adj = knn_graph(f, 1)
        >>> print(adj)
        (0, 1)  1.0
        (1, 0)  1.0
        (2, 1)  1.0
    """
    if not isinstance(feat, np.ndarray):
        raise TypeError("The feat type is {},\
                        but it should be numpy.ndarray.".format(type(feat)))
    if not isinstance(k, int):
        raise TypeError("The k type is {},\
                        but it should be int.".format(type(k)))
    size, dim = feat.shape

    feat = feat.astype('float32')
    index = faiss.IndexFlat(dim)
    index.add(feat)

    if not loop:
        k += 1
    if gpu:
        res = faiss.StandardGpuResources()
        faiss.index_cpu_to_gpu(res, index=index, device=device)
    d, nbrs = index.search(feat, k)
    if dis:
        col = np.array([i for i in range(len(nbrs)) for j in d[i] if j < dis])
        row = np.array([j for i in d for j in i if j < dis])
    else:
        col = np.array([i for i in range(len(nbrs)) for _ in range(len(nbrs[i]))])
        row = np.array([j for i in nbrs for j in i])
    if not loop:
        mask = np.argwhere((row - col) != 0)
        row = np.squeeze(row[mask])
        col = np.squeeze(col[mask])
    data = np.ones(len(row))
    g = sp.csr_matrix((data, (col, row)), shape=(size, size)).tocoo()
    return g


def distance(node_feat, edge_index, norm: bool = True, max_value=None):
    r"""
    In a graph data structure, assign the Euclidean distance of connected nodes as an attribute to edges

    .. note:: The input feature should be float16, float32.

    Args:
        node_feat(Tensor):node feature, shape:(N, F)
        edge_index(Tensor):edge index, shape:(2, col)
        norm(bool):Normalized
        max_value(float):normalized maximum

    Returns:
        coo, adjacency matrix

    Examples:
        >>> import numpy as np
        >>> import scipy.sparse as sp
        >>> from mindspore_gl.sampling import distance
        >>> node_feat = ms.Tensor([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])
        >>> edge_index = ms.Tensor([[0, 1], [2, 2]])
        >>> adj = distance(node_feat, edge_index)
        >>> print(adj)
        (0, 2)  1.0
        (1, 2)  0.5
    """
    if not isinstance(node_feat, ms.Tensor):
        raise TypeError("The node_feat type is {},\
                        but it should be Tensor.".format(type(node_feat)))
    if not isinstance(edge_index, ms.Tensor):
        raise TypeError("The edge_index type is {},\
                        but it should be Tensor.".format(type(edge_index)))

    col_feat = ms.ops.Gather()(node_feat, edge_index[0], 0)
    row_feat = ms.ops.Gather()(node_feat, edge_index[1], 0)
    data = col_feat - row_feat
    dist = ms.nn.Norm(-1)(data.astype(ms.float32))
    if norm:
        dist = dist / (ms.ops.ReduceMax()(dist) if max_value is None else max_value)

    adj_coo = sp.csr_matrix((dist.asnumpy(), (edge_index[0], edge_index[1]))\
                            , shape=(3, 3)).tocoo()
    return adj_coo
