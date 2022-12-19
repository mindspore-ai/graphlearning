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
"""random walks on graphs"""
import numpy as np
from mindspore_gl.graph.graph import MindHomoGraph
from mindspore_gl import sample_kernel

__all__ = ['random_walk_unbias_on_homo']


def random_walk_unbias_on_homo(homo_graph: MindHomoGraph,
                               seeds: np.ndarray,
                               walk_length: int):
    """
    random walks on homo graph.

    Args:
        homo_graph(MindHomoGraph): the source graph which is sampled from
        seeds(np.ndarray) : random seeds for sampling
        walk_length(int): sample path length

    Returns:
        - array, sample node :math:`(len(seeds), walk_length)`

    Raises:
        TypeError: If `walk_length` is not a positive integer.
        TypeError: If `seeds` is not numpy.ndarray int 32.

    Supported Platforms:
        ``GPU`` ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import networkx
        >>> from scipy.sparse import csr_matrix
        >>> from mindspore_gl.graph.graph import MindHomoGraph, CsrAdj
        >>> from mindspore_gl.sampling.randomwalks import random_walk_unbias_on_homo
        >>> node_count = 10000
        >>> edge_prob = 0.1
        >>> graph = networkx.generators.random_graphs.fast_gnp_random_graph(node_count, edge_prob)
        >>> edge_array = np.transpose(np.array(list(graph.edges)))
        >>> row = edge_array[0]
        >>> col = edge_array[1]
        >>> data = np.zeros(row.shape)
        >>> csr_mat = csr_matrix((data, (row, col)), shape=(node_count, node_count))
        >>> generated_graph = MindHomoGraph()
        >>> node_dict = {idx: idx for idx in range(node_count)}
        >>> edge_count = col.shape[0]
        >>> edge_ids = np.array(list(range(edge_count))).astype(np.int32)
        >>> generated_graph.set_topo(CsrAdj(csr_mat.indptr.astype(np.int32), csr_mat.indices.astype(np.int32)),
        ... node_dict, edge_ids)
        >>> nodes = np.arange(0, node_count)
        >>> out = random_walk_unbias_on_homo(homo_graph=generated_graph, seeds=nodes[:5].astype(np.int32),
        ... walk_length=10)
        >>> print(out)
        # results will be random for suffle
        [[   0 9493 8272 1251 3922 4180  211 1083 4198 9981 7669]
         [   1 1585 1308 4703 1115 4989 9365 1098 1618 5987 8312]
         [   2 2352 7214 5956 2184 1573 1352 7005 2325 6211 8667]
         [   3 8723 5645 3691 4857 5501  113 4140 6666 2282 1248]
         [   4 4354 9551 5224 3156 8693  346 8899 6046 6011 5310]]
    """

    if not isinstance(seeds, np.ndarray):
        raise TypeError("The positive data type is {},\
                        but it should be ndarray or list.".format(type(seeds)))
    if not isinstance(walk_length, int) or walk_length <= 0:
        raise TypeError("The node type is {},\
                        but it should be positive int.".format(type(walk_length)))

    # sample
    out = sample_kernel.random_walk_cpu_unbias(homo_graph.adj_csr.indptr,
                                               homo_graph.adj_csr.indices,
                                               walk_length, seeds)
    return out
