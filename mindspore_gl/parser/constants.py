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
"""Constants"""
SRC_IDX = "src_idx"
DST_IDX = "dst_idx"
VER_SUBGRAPH_IDX = "ver_subgraph_idx"
EDGE_SUBGRAPH_IDX = "edge_subgraph_idx"
GRAPH_MASK = "graph_mask"
N_NODES = "n_nodes"
N_EDGES = "n_edges"
N_GRAPHS = "n_graphs"
SCATTER_SRC_IDX = "scatter_src_idx"
SCATTER_DST_IDX = "scatter_dst_idx"
SCATTER_VER_SUBGRAPH_IDX = "scatter_ver_subgraph_idx"
SCATTER_EDGE_SUBGRAPH_IDX = "scatter_edge_subgraph_idx"
GATHER_OP = "GATHER"
SCATTER_ADD_OP = "SCATTER_ADD"
SCATTER_MAX_OP = "SCATTER_MAX"
SCATTER_MIN_OP = "SCATTER_MIN"
SHAPE_OP = "SHAPE"
RESHAPE_OP = "RESHAPE"
ZEROS_OP = "ZEROS"
FILL_OP = "FILL"
MASKED_FILL_OP = "MASKED_FILL"
IS_INF_OP = "IS_INF"
BACKEND_NAME = None
INDICES = "indices"
INDPTR = "indptr"
INDICES_BACKWARD = "indices_backward"
INDPTR_BACKWARD = "indptr_backward"

CSR_REDUCE_SUM_OP = 'CSR_REDUCE_SUM'
GRAPH_FIELD_NAMES = [SRC_IDX, DST_IDX, N_NODES, N_EDGES]
CSR_GRAPH_FIELD_NAMES = [INDICES, INDPTR, N_NODES, N_EDGES, INDICES_BACKWARD, INDPTR_BACKWARD]
BATCHED_GRAPH_FIELD_NAMES = [SRC_IDX, DST_IDX, N_NODES, N_EDGES,
                             VER_SUBGRAPH_IDX, EDGE_SUBGRAPH_IDX, GRAPH_MASK]
BATCHED_CSR_GRAPH_FIELD_NAMES = [INDICES, INDPTR, N_NODES, N_EDGES, INDICES_BACKWARD, INDPTR_BACKWARD,
                                 VER_SUBGRAPH_IDX, EDGE_SUBGRAPH_IDX, GRAPH_MASK]
