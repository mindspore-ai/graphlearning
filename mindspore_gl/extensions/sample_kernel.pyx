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

import numpy as np
cimport numpy as np
cimport cython
from libcpp.map cimport map
from libcpp.set cimport set
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libc.stdlib cimport rand, RAND_MAX
from libcpp cimport bool
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
def map_edges(np.ndarray[np.int32_t, ndim=2] edges, reindex):
    """Mapping edges by given map dictionary
    """
    cdef unordered_map[int, int] map_dict = reindex
    cdef int i = 0
    cdef int num_edges = edges.shape[1]
    cdef int [:, :] edges_new = edges
    with nogil:
        for i in prange(num_edges, schedule="static"):
            edges_new[0, i] = map_dict[edges_new[0, i]]
            edges_new[1, i] = map_dict[edges_new[1, i]]
    return edges_new


@cython.boundscheck(False)
@cython.wraparound(False)
def map_nodes(nodes, reindex):
    """Mapping node id by given map dictionary
    """
    cdef np.ndarray[np.int32_t, ndim=1] t_nodes = np.array(nodes, dtype=np.int32)
    cdef unordered_map[int, int] map_dict = reindex
    cdef int i = 0
    cdef int num_nodes = len(nodes)
    cdef np.ndarray[np.int32_t, ndim=1] nodes_new = np.zeros([num_nodes], dtype=np.int32)
    with nogil:
        for i in xrange(num_nodes):
            nodes_new[i] = map_dict[t_nodes[i]]
    return nodes_new

@cython.boundscheck(False)
@cython.wraparound(False)
def sample_one_hop_unbias(np.ndarray[np.int32_t, ndim=1] csr_row, np.ndarray[np.int32_t, ndim=1] csr_col, int neighbor_num,
                          np.ndarray[np.int32_t, ndim=1] seeds, bool replace=False):

    
    cdef int seeds_length = len(seeds)

    cdef int seed_idx
    cdef int node
    cdef int col_start, col_end
    cdef int total_edge_num = 0
    cdef int offset = 0
    cdef int idx

    for seed_idx in xrange(seeds_length):
        node = seeds[seed_idx]
        col_start, col_end = csr_row[node], csr_row[node + 1]
        if col_end - col_start <= neighbor_num:
            total_edge_num += col_end - col_start
        else:
            total_edge_num += neighbor_num
    #
    cdef np.ndarray[np.int32_t, ndim=1] edge_ids = np.zeros([total_edge_num], dtype=np.int32)

    # define result
    cdef np.ndarray[np.int32_t, ndim=2] res_edge_index = np.zeros([2, total_edge_num], dtype=np.int32)
    for seed_idx in xrange(seeds_length):
        node = seeds[seed_idx]
        col_start, col_end = csr_row[node], csr_row[node + 1]
        if col_end - col_start <= neighbor_num:
            # print("neighbor < count")
            res_edge_index[0][offset: offset + col_end - col_start] = node

            res_edge_index[1][offset: offset + col_end - col_start] = csr_col[col_start: col_end]
            for idx in xrange(col_end - col_start):
                edge_ids[offset + idx] = col_start + idx
            offset += col_end - col_start
        else:

            choose_index(edge_ids, neighbor_num, col_end - col_start, offset, col_start)
            #update
            res_edge_index[0, offset: offset + neighbor_num] = node
            res_edge_index[1, offset: offset + neighbor_num] = csr_col[edge_ids[offset: offset + neighbor_num]]
            offset += neighbor_num

    return res_edge_index, edge_ids


@cython.boundscheck(False)
@cython.wraparound(False)
def set_node_map_idx(np.ndarray[ndim=1, dtype=np.int32_t] node_map_idx, np.ndarray[ndim=1, dtype=np.int32_t] graph_nodes):
    cdef int [:] node_map_idx_view = node_map_idx
    cdef int [:] graph_nodes_view = graph_nodes
    cdef int node_count = graph_nodes.shape[0]
    cdef int idx
    with nogil:
        for idx in prange(node_count, schedule="static"):
            node_map_idx_view[graph_nodes_view[idx]: graph_nodes_view[idx + 1]] = idx
    return node_map_idx

@cython.boundscheck(False)
@cython.wraparound(False)
def set_edge_map_idx(np.ndarray[ndim=1, dtype=np.int32_t] edge_map_idx, np.ndarray[ndim=1, dtype=np.int32_t] graph_edges):
    return set_node_map_idx(edge_map_idx, graph_edges)

@cython.boundscheck(False)
@cython.wraparound(False)
def choose_index(np.ndarray[ndim=1, dtype=np.int32_t] rnd, int neighbor_count, int total_neighbor_size,
                 int offset, int col_start):
    # Sample without replacement via Robert Floyd algorithm
    # https://www.nowherenearithaca.com/2013/05/robert-floyds-tiny-and-beautiful.html
    cdef unordered_set[int] perm
    cdef int j, t
    cdef idx = 0
    for j in xrange(total_neighbor_size - neighbor_count, total_neighbor_size):
        t = rand() % j
        if perm.find(t) != perm.end():
            perm.insert(t)
            rnd[offset + idx] = t + col_start
        else:
            perm.insert(j)
            rnd[offset + idx] = j + col_start
        idx += 1


@cython.wraparound(False)
@cython.boundscheck(False)
def random_walk_cpu_unbias(np.ndarray[np.int32_t, ndim=1] csr_row, np.ndarray[np.int32_t, ndim=1] csr_col,
                           int walk_length, np.ndarray[np.int32_t, ndim=1] seeds, int default_value = -1):
    cdef int seeds_length = seeds.shape[0]
    cdef np.ndarray[np.int32_t, ndim=2] out = np.full([seeds_length, walk_length + 1], -1, dtype=np.int32)
    cdef int idx
    cdef int node
    cdef int row_start, row_end
    cdef int cur_ptr = 0
    cdef int sampled_node
    for idx in xrange(seeds_length):
        node = seeds[idx]
        out[idx][0] = node
        sample_node = node
        for cur_ptr in xrange(walk_length):
            row_start = csr_row[sample_node]
            row_end = csr_row[sample_node + 1]
            sampled_node = csr_col[row_start + rand() % (row_end - row_start)]
            out[idx][cur_ptr + 1] = sampled_node

    return out
    
@cython.boundscheck(False)
@cython.wraparound(False)
def skip_gram_gen_pair(vector[long long] walk, long win_size=5):
    cdef vector[long long] s_ids
    cdef vector[long long] d_ids
    cdef long long steps = len(walk)
    cdef long long real_win_size, left, right, i
    cdef np.ndarray[np.int64_t, ndim=1] rnd = np.random.randint(1,  win_size+1,
                                    dtype=np.int64, size=steps)
    with nogil:
        for i in xrange(steps):
            real_win_size = rnd[i]
            left = i - real_win_size
            if left < 0:
                left = 0
            right = i + real_win_size
            if right >= steps:
                right = steps - 1
            for j in xrange(left, right+1):
                if walk[i] == walk[j]:
                    continue
                s_ids.push_back(walk[i])
                d_ids.push_back(walk[j])
    return s_ids, d_ids

def node2vec_random_walk(np.ndarray[np.int32_t, ndim=1] csr_row, np.ndarray[np.int32_t, ndim=1] csr_col,
                         int walk_length, np.ndarray[np.int32_t, ndim=1] seeds, float p, float q):
    """
    Generate random walk traces from an array of starting nodes based on the node2vec model.
    Paper: `node2vec: Scalable Feature Learning for Networks
    <https://arxiv.org/abs/1607.00653>`__.
    The returned traces all have length ``walk_length + 1``, where the first node
    is the starting node itself.
    Note that if a random walk stops in advance, We pads the trace with -1 to have the same
    length.
    """





