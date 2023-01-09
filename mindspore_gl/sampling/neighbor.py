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
"""Sampling neighbor"""
from typing import List
import numpy as np
import mindspore_gl
from mindspore_gl.graph import MindHomoGraph
from mindspore_gl import sample_kernel


def map_edge_index(layered_edges, reindex_dict):
    for layer_index, layer_edge in enumerate(layered_edges):
        layered_edges[layer_index] = sample_kernel.map_edges(layer_edge, reindex_dict)
    return layered_edges


def sage_sampler_on_homo(homo_graph: mindspore_gl.graph.MindHomoGraph, seeds, neighbor_nums: List[int]):
    """
    GraphSage sampling on MindHomoGraph.

    Args:
        homo_graph(mindspore_gl.graph.MindHomoGraph): input graph.
        seeds(numpy.ndarray): start nodes for neighbor sampling.
        neighbor_nums(List): neighbor nums for each hop.

    Returns:
        - **layered_edges_{idx}** (numpy.array) - edge array for hop idx.
        - **layered_eids_{idx}** (numpy.array) - edge id array for hop idx.
        - **all_nodes** - all nodes' global ids.
        - **seeds_idx** - seeds local ids.

    Raises:
        TypeError: If 'homo_graph' is not a MindHomoGraph class.
        TypeError: If 'seeds' is not a numpy.ndarray.
        TypeError: If 'neighbor_nums' is not a list.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import networkx
        >>> import numpy as np
        >>> from scipy.sparse import csr_matrix
        >>> from mindspore_gl.graph import MindHomoGraph, CsrAdj
        >>> from mindspore_gl.sampling.neighbor import sage_sampler_on_homo
        >>> node_count = 10
        >>> edge_prob = 0.3
        >>> graph = networkx.generators.random_graphs.fast_gnp_random_graph(node_count, edge_prob, seed=1)
        >>> edge_array = np.transpose(np.array(list(graph.edges)))
        >>> row = edge_array[0]
        >>> col = edge_array[1]
        >>> data = np.zeros(row.shape)
        >>> csr_mat = csr_matrix((data, (row, col)), shape=(node_count, node_count))
        >>> generated_graph = MindHomoGraph()
        >>> node_dict = {idx: idx for idx in range(node_count)}
        >>> edge_count = col.shape[0]
        >>> edge_ids = np.array(list(range(edge_count))).astype(np.int32)
        >>> generated_graph.set_topo(CsrAdj(csr_mat.indptr.astype(np.int32), csr_mat.indices.astype(np.int32)),\
        ... node_dict, edge_ids)
        >>> nodes = np.arange(0, node_count)
        >>> res = sage_sampler_on_homo(homo_graph=generated_graph, seeds=nodes[:3].astype(np.int32),\
        ... neighbor_nums=[2, 2])
        >>> print(res)
        {'seeds_idx': array([0, 3, 2], dtype=int32), 'all_nodes': array([0, 1, 2, 1, 4, 5, 6, 5, 6, 7, 8, 9],
           dtype=int32), 'layered_edges_0': array([[0, 0, 3, 3, 2],
           [3, 4, 7, 8, 7]], dtype=int32), 'layered_eids_0': array([[0, 0, 3, 3, 2],
           [3, 4, 7, 8, 7]], dtype=int32), 'layered_edges_1': array([[ 3,  3,  4,  4,  7,  8,  8],
           [ 7,  8, 10, 11,  9, 10, 11]], dtype=int32), 'layered_eids_1': array([[ 3,  3,  4,  4,  7,  8,  8],
           [ 7,  8, 10, 11,  9, 10, 11]], dtype=int32)}

    """
    if not isinstance(homo_graph, MindHomoGraph):
        raise TypeError("For sage_sampler_on_homo, the 'homo_graph' must a MindHomoGraph, but got "
                        f"{type(homo_graph).__name__}.")
    if not isinstance(seeds, np.ndarray):
        raise TypeError("For sage_sampler_on_homo, the 'seeds' must a numpy array, but got "
                        f"{type(seeds).__name__}.")
    if not isinstance(neighbor_nums, list):
        raise TypeError("For sage_sampler_on_homo, the 'seeds' must a list, but got "
                        f"{type(neighbor_nums).__name__}.")
    saved_seeds = seeds
    all_nodes = [seeds]
    layered_edges = []
    layered_eids = []
    for neighbor_num in neighbor_nums:
        edge_index, edge_ids = sample_kernel.sample_one_hop_unbias(homo_graph.adj_csr.indptr,
                                                                   homo_graph.adj_csr.indices,
                                                                   neighbor_num, seeds)
        layered_edges.append(edge_index)
        layered_eids.append(edge_ids)
        seeds = np.unique(edge_index[1])

        all_nodes.append(seeds)

    # get_node_features:
    all_nodes = np.concatenate(all_nodes, axis=0)

    # reindex sampled result
    reindex_dict = {all_nodes[index]: index for index in range(all_nodes.shape[0])}
    layered_edges = map_edge_index(layered_edges, reindex_dict)

    seeds_idx = np.zeros(saved_seeds.shape, dtype=np.int32)
    for idx in range(saved_seeds.shape[0]):
        seeds_idx[idx] = reindex_dict[saved_seeds[idx]]
    res = {
        "seeds_idx": seeds_idx,
        "all_nodes": all_nodes,
    }
    for layer_idx, layer in enumerate(layered_edges):
        res[f'layered_edges_{layer_idx}'] = layer
        res[f'layered_eids_{layer_idx}'] = layer
    return res
