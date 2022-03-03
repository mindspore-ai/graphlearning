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

"""Computes the k-hop subgraph around a subset of nodes"""
import numpy as np
#pylint:disable=E1123


def gen_edge_lookup(adj_coo, node_count, flow='source_to_target'):
    """
    Generate edge lookup for each node.

    Note:
        The returned lookup can be saved and reused as long as the graph's topology doesn't change.

    Args:
        adj_coo(numpy.ndarray): input adj of graph.
        node_count(int): the number of nodes.
        flow: the visit direction.

    Returns:
        - **begins** (numpy.ndarray) - beginning positions in indices of each node, in length of no. of nodes plus one.
        - **indices** (numpy.ndarray) - edges indices, in length of no. of edges.
    """
    if flow == 'target_to_source':
        local = adj_coo[0]
    else:
        local = adj_coo[1]

    edge_count = adj_coo.shape[1]
    begins = np.zeros(node_count + 1, dtype=int)
    indices = np.zeros(edge_count, dtype=int)
    per_node_edge_counts = np.zeros(node_count + 1, dtype=int)

    for node_idx in local:
        per_node_edge_counts[node_idx] += 1

    pos = 0
    for node_idx in range(begins.shape[0]):
        begins[node_idx] = pos
        pos += per_node_edge_counts[node_idx]

    per_node_edge_counts.fill(0)
    for edge_idx, node_idx in enumerate(local):
        pos = begins[node_idx] + per_node_edge_counts[node_idx]
        indices[pos] = edge_idx
        per_node_edge_counts[node_idx] += 1

    return begins, indices


def fast_k_hop_subgraph(edge_lookup, node_idx, num_hops, adj_coo, relabel_nodes=False, flow='source_to_target'):
    """
    Fast k-hop sampling on MindHomoGraph.

    Note:
        The time complexity doesn't depend on the graph size. The returned subgraph will be slightly different from
        k_hop_subgraph() that edges connecting to subgraph's nodes but exceed the hopping distance won't be returned.

    Args:
        edge_lookup(tuple[numpy.ndarray, numpy.ndarray]): edge lookup returned by gen_edge_lookup().
        node_idx(int, list, tuple or numpy.ndarray): sampling subgraph around 'node_idx'.
        num_hops(int): sampling 'num_hops' hop subgraph.
        adj_coo(numpy.ndarray): input adj of graph.
        relabel_nodes(bool): node indexes need relabel or not.
        flow: the visit direction.

    Returns:
        res(dict), has 4 keys 'subset', 'edge_subset', 'adj_coo', 'inv', where,

        - **subset** (numpy.ndarray) - nodes' idx of sampled K-hop subgraph.
        - **edge_subset** (numpy.ndarray) - edges' idx of sampled K-hop subgraph.
        - **adj_coo** (numpy.ndarray) - adj of sampled K-hop subgraph.
        - **inv** (list) - the mapping from node indices in `node_idx` to their new location.
    """
    if flow == 'target_to_source':
        remote = adj_coo[1]
    else:
        remote = adj_coo[0]

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = np.array([node_idx]).flatten()

    lookup_begins, lookup_idxs = edge_lookup

    node_idxs = node_idx
    node_subsets = [node_idxs]
    edge_subsets = []
    for _ in range(num_hops):
        new_node_idxs = []
        for node in node_idxs:
            if lookup_begins[node] >= lookup_begins[node + 1]:
                continue
            edges = lookup_idxs[lookup_begins[node]:lookup_begins[node + 1]]
            edge_subsets.append(edges)
            new_node_idxs.append(remote[edges])
        node_idxs = np.concatenate(new_node_idxs)
        if node_idxs.shape[0] == 0:
            break
        node_subsets.append(node_idxs)

    node_subset, inv = np.unique(np.concatenate(node_subsets), return_inverse=True)
    inv = inv[:node_idx.size]
    edge_subset = np.unique(np.concatenate(edge_subsets))

    subgraph_adj = adj_coo[:, edge_subset]

    if relabel_nodes:
        node_map = dict(((o, n) for n, o in enumerate(node_subset)))
        for i in range(subgraph_adj.shape[0]):
            for j in range(subgraph_adj.shape[1]):
                subgraph_adj[i, j] = node_map[subgraph_adj[i, j]]

    res = {"subset": node_subset, "edge_subset": edge_subset, "adj_coo": subgraph_adj, "inv": inv}
    return res


def k_hop_subgraph(node_idx, num_hops, adj_coo, node_count, relabel_nodes=False, flow='source_to_target'):
    """
    K-hop sampling on MindHomoGraph

    Args:
        node_idx(int, list, tuple or numpy.ndarray): sampling subgraph around 'node_idx'.
        num_hops(int): sampling 'num_hops' hop subgraph.
        adj_coo(numpy.ndarray): input adj of graph.
        node_count(int): the number of nodes.
        relabel_nodes(bool): node indexes need relabel or not.
        flow: the visit direction.

    Returns:
        res(dict), has 4 keys 'subset', 'adj_coo', 'inv', 'edge_mask', where,

        - **subset** (numpy.ndarray) - nodes' idx of sampled K-hop subgraph.
        - **adj_coo** (numpy.ndarray) - adj of sampled K-hop subgraph.
        - **inv** (list) - the mapping from node indices in `node_idx` to their new location.
        - **edge_mask** (numpy.ndarray) - the edge mask indicating which edges were preserved.
    """
    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = adj_coo
    else:
        col, row = adj_coo

    node_mask = np.empty_like(row, shape=node_count, dtype=np.bool_)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = np.array([node_idx]).flatten()

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill(False)
        node_mask[subsets[-1]] = True
        edge_mask = np.take(node_mask, row)
        subsets.append(col[edge_mask])

    subsets = np.concatenate(subsets)
    subset, inv = np.unique(subsets, return_inverse=True)

    inv = inv[:node_idx.size]

    node_mask.fill(False)
    node_mask[subset] = True

    edge_mask = node_mask[row] & node_mask[col]

    adj_coo = adj_coo[:, edge_mask]

    if relabel_nodes:
        node_idx = np.full((node_count,), -1)
        node_idx[subset] = np.arange(subset.shape[0])
        adj_coo = node_idx[adj_coo]

    res = {"subset": subset, "adj_coo": adj_coo, "inv": inv, "edge_mask": edge_mask}
    return res
