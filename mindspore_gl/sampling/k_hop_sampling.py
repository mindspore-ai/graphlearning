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
