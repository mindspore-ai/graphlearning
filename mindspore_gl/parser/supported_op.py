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
"""Supported operations."""
from collections import namedtuple
from .constants import SCATTER_ADD_OP, SCATTER_MAX_OP, SCATTER_MIN_OP, VER_SUBGRAPH_IDX, EDGE_SUBGRAPH_IDX, \
    SCATTER_DST_IDX, SCATTER_SRC_IDX, SCATTER_VER_SUBGRAPH_IDX, SCATTER_EDGE_SUBGRAPH_IDX, N_NODES, N_GRAPHS

OpInfos = namedtuple("OpInfos", ["func_name", "args"])

supported_ops = {
    # Graph/BatchedGraph supported operations.

    "sum": OpInfos("transform_agg_func", (SCATTER_ADD_OP, False)),
    "max": OpInfos("transform_agg_func", (SCATTER_MAX_OP, False)),
    "min": OpInfos("transform_agg_func", (SCATTER_MIN_OP, False)),
    "avg": OpInfos("transform_agg_func", (SCATTER_ADD_OP, True)),

    "dot": OpInfos("transform_dot_func", ()),
    "topk_nodes": OpInfos("transform_topk_func", ()),
    "topk_edges": OpInfos("transform_topk_func", ()),

    "node_mask": OpInfos("transform_get_mask_func", (VER_SUBGRAPH_IDX,)),
    "edge_mask": OpInfos("transform_get_mask_func", (EDGE_SUBGRAPH_IDX,)),

    "in_degree": OpInfos("transform_scatter_idx_func", (N_NODES, SCATTER_DST_IDX)),
    "out_degree": OpInfos("transform_scatter_idx_func", (N_NODES, SCATTER_SRC_IDX)),
    "num_of_nodes": OpInfos("transform_scatter_idx_func", (N_GRAPHS, SCATTER_VER_SUBGRAPH_IDX)),
    "num_of_edges": OpInfos("transform_scatter_idx_func", (N_GRAPHS, SCATTER_EDGE_SUBGRAPH_IDX)),

    "adj_to_dense": OpInfos("transform_adj_to_dense_func", ()),

    "sum_nodes": OpInfos("transform_readout_func", (SCATTER_ADD_OP, SCATTER_VER_SUBGRAPH_IDX, False)),
    "sum_edges": OpInfos("transform_readout_func", (SCATTER_ADD_OP, SCATTER_EDGE_SUBGRAPH_IDX, False)),
    "max_nodes": OpInfos("transform_readout_func", (SCATTER_MAX_OP, SCATTER_VER_SUBGRAPH_IDX, False)),
    "max_edges": OpInfos("transform_readout_func", (SCATTER_MAX_OP, SCATTER_EDGE_SUBGRAPH_IDX, False)),
    "avg_nodes": OpInfos("transform_readout_func", (SCATTER_ADD_OP, SCATTER_VER_SUBGRAPH_IDX, True)),
    "avg_edges": OpInfos("transform_readout_func", (SCATTER_ADD_OP, SCATTER_EDGE_SUBGRAPH_IDX, True)),
    "softmax_nodes": OpInfos("transform_readout_softmax_func", (SCATTER_VER_SUBGRAPH_IDX, VER_SUBGRAPH_IDX)),
    "softmax_edges": OpInfos("transform_readout_softmax_func", (SCATTER_EDGE_SUBGRAPH_IDX, EDGE_SUBGRAPH_IDX)),
    "broadcast_nodes": OpInfos("transform_readout_broadcast_func", (VER_SUBGRAPH_IDX,)),
    "broadcast_edges": OpInfos("transform_readout_broadcast_func", (EDGE_SUBGRAPH_IDX,)),
    "batched_topk_nodes": OpInfos("transform_readout_topk_func", (VER_SUBGRAPH_IDX,)),
    "batched_topk_edges": OpInfos("transform_readout_topk_func", (EDGE_SUBGRAPH_IDX,)),

    "get_homo_graph": OpInfos("transform_get_homo_func", ()),
}
