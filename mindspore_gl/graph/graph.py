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

"""Claesses for Graph data structure"""
from typing import Dict
from collections import namedtuple

import numpy as np
import mindspore_gl.sample_kernel as kernel

CsrAdj = namedtuple("csr_adj", ['indptr', 'indices'])


def coo_to_csr():
    raise NotImplementedError()


def csr_to_coo(csr: CsrAdj):
    coo_adj = [[], []]
    j = 0
    for i in range(len(csr.indptr) - 1):
        while j < csr.indptr[i + 1]:
            coo_adj[0].append(i)
            coo_adj[1].append(csr.indices[j])
            j += 1
    return coo_adj


class MindRelationGraph:
    """
    Relation Graph, a simple implementation of relation graph structure in mindspore-gl.

    Args:
        src_node_type(str): source node type.
        dst_node_type(str): destination node type.
        edge_type(str): edge type.

    """
    def __init__(self, src_node_type: str, dst_node_type: str,
                 edge_type: str):
        self._u_type = src_node_type
        self._v_type = dst_node_type
        self._e_type = edge_type
        self._relation_type = f"{src_node_type}_{edge_type}_{dst_node_type}"

        # dataloader
        self._adj_csr = None
        self._adj_coo = None
        self._reverse_adj_csr = None
        self._node_dict = None
        self._node_ids = None
        self._edge_ids = None

    ################################
    # Initialize Graph
    ################################

    def set_topo(self, adj_csr: CsrAdj, node_dict=None, edge_ids: np.ndarray = None):
        """
        set topology for relation graph by either csr_adj.

        Args:
            adj_csr(csr_adj): csr format description of adjacent matrix.
            edge_ids(numpy.ndarray): edge ids for each edge.
            node_dict(Dict): global->local node id.
        """
        self._adj_csr = adj_csr
        self._node_dict = node_dict
        self._node_ids = None if node_dict is None else np.array(list(node_dict.keys()))
        self._edge_ids = edge_ids


    #####################################
    # Query Graph, With Lazy Computation
    #####################################
    def has_node(self, node):
        """
        If this relation graph has certain node.

        Args:
            node: global node id

        Returns:
            bool, indicate if node is in this graph

        """
        return node < self._adj_csr.indptr.shape[0] \
               if self._node_dict is None else self._node_dict.get(node) is not None

    def successors(self, src_node):
        mapped_idx = self._node_dict.get(src_node, None)
        assert mapped_idx is not None
        neighbor_start = self._adj_csr.indptr[mapped_idx]
        neighbor_end = self._adj_csr.indptr[mapped_idx + 1]
        neighbors = self._adj_csr.indices[neighbor_start: neighbor_end]
        node_ids = neighbors if self._node_ids is None else self._node_ids[neighbors]
        return node_ids

    def predecessors(self, dst_node):
        pass

    def out_degree(self, src_node):
        mapped_idx = src_node if self._node_dict is None else self._node_dict.get(src_node, None)
        assert mapped_idx is not None
        neighbor_start = self._adj_csr.indptr[mapped_idx]
        neighbor_end = self._adj_csr.indptr[mapped_idx + 1]
        return neighbor_end - neighbor_start

    def out_degrees(self, src_nodes):
        pass

    def in_degree(self, dst_node):
        pass

    def in_degrees(self, dst_nodes):
        pass

    def format(self, out_format):
        pass


    #########################
    # properties
    #########################
    @property
    def node_num(self):
        return self._adj_csr.indptr.shape[0]

    @property
    def edge_num(self):
        return self._adj_csr.indices.shape[0]

    @property
    def relation_type(self):
        return self._relation_type

    @property
    def nodes(self):
        return np.arange(self.node_num) if self._node_ids is None else self._node_ids

    @property
    def edges(self):
        return np.arange(self.edge_num) if self._edge_ids is None else self._edge_ids

    @property
    def adj_csr(self) -> CsrAdj:
        return self._adj_csr

    ######################################
    # Transform Graph To Different Format
    ######################################


class BatchMeta:
    """
    BatchMeta, meta information for a batched graph.

    Args:
        graph_nodes(numpy.array): accumulated node sum for graphs in batched graph(first element is 0).
        graph_edges(numpy.array): accumulated edge sum for graphs in batched graph(first element is 0).

    """
    def __init__(self, graph_nodes, graph_edges):
        self._graph_nodes = graph_nodes
        self._graph_edges = graph_edges

        #######################
        # For Lazy Computation
        #######################
        self._node_map_idx = None
        self._edge_map_idx = None

    @property
    def graph_nodes(self):
        """
        Returns:
            numpy.array, accumulated node sum for graphs in batched graph(first element is 0)

        """
        return self._graph_nodes

    @property
    def graph_edges(self):
        """
        Returns:
            numpy.array, accumulated edge sum for graphs in batched graph(first element is 0)
        """
        return self._graph_edges

    @property
    def graph_count(self):
        """
        Returns:
            int, total graph count in this batched graph

        """
        return self._graph_edges.shape[0] - 1

    @property
    def node_map_idx(self):
        """
        Returns:
            numpy.array, array indicate graph index for each node

        """
        if self._node_map_idx is not None:
            return self._node_map_idx
        self._node_map_idx = np.zeros([self._graph_nodes[-1]], dtype=np.int32)
        kernel.set_node_map_idx(self._node_map_idx, self._graph_nodes)
        return self._node_map_idx

    @property
    def edge_map_idx(self):
        """
        Returns:
            numpy.array, array indicate graph index for each edge
        """
        if self._edge_map_idx is not None:
            return self._edge_map_idx
        self._edge_map_idx = np.zeros([self._graph_edges[-1]], dtype=np.int32)
        kernel.set_edge_map_idx(self._edge_map_idx, self._graph_edges)
        return self._edge_map_idx

    def __getitem__(self, graph_idx):
        """
        return node count and edge count for idx graph

        Args:
            graph_idx(int): graph idx for query

        Returns:
            (int, int), (node_count, edge_count)
        """
        assert graph_idx < self.graph_count, "index out of range"
        return (self.graph_nodes[graph_idx + 1] - self.graph_nodes[graph_idx],
                self.graph_edges[graph_idx + 1] - self.graph_edges[graph_idx])


class MindHomoGraph:
    """
    in-memory homo graph, edge_type == 1
    """
    def __init__(self):

        self._node_dict = None
        self._node_ids = None
        self._edge_ids = None

        self._adj_csr: CsrAdj = None
        self._adj_coo = None

        ################
        #
        ################
        self._node_count = 0
        self._edge_count = 0

        ###################
        # Batch Meta Info
        ###################
        self._batch_meta = None

    ############################################
    # initialize Graph
    ###########################################
    def set_topo(self, adj_csr: np.ndarray, node_dict, edge_ids: np.ndarray):
        self._adj_csr = adj_csr
        self._node_dict = node_dict
        self._edge_ids = edge_ids
        self._node_ids = np.array(list(node_dict.keys()))

    def set_topo_coo(self, adj_coo, node_dict=None, edge_ids: np.ndarray = None):
        self._adj_coo = adj_coo
        self._node_dict = node_dict
        self._node_ids = None if node_dict is None else np.array(list(node_dict.keys()))
        self._edge_ids = edge_ids

    ##########################################
    # Query With Lazy Computation
    ##########################################
    def neighbors(self, node):
        self._check_csr()
        mapped_idx = self._node_dict.get(node, None)
        assert mapped_idx is not None
        neighbor_start = self._adj_csr.indptr[mapped_idx]
        neighbor_end = self._adj_csr.indptr[mapped_idx + 1]
        neighbors = self._adj_csr.indices[neighbor_start: neighbor_end]
        node_ids = self._node_ids[neighbors]
        return node_ids

    def degree(self, node):
        self._check_csr()
        mapped_idx = self._node_dict.get(node, None)
        assert mapped_idx is not None
        neighbor_start = self._adj_csr.indptr[mapped_idx]
        neighbor_end = self._adj_csr.indptr[mapped_idx + 1]
        return neighbor_end - neighbor_start

    @property
    def adj_csr(self):
        self._check_csr()
        return self._adj_csr

    @property
    def adj_coo(self):
        self._check_coo()
        return self._adj_coo

    @adj_coo.setter
    def adj_coo(self, adj_coo):
        del self._adj_csr
        self._adj_csr = None
        self._adj_coo = adj_coo

    @property
    def edge_count(self):
        """Edge count of graph"""
        if self._edge_count > 0:
            return self._edge_count

        if self._adj_csr is not None:
            self._edge_count = self._adj_csr.indices.shape[0]
        if self._adj_coo is not None:
            self._edge_count = self._adj_coo.shape[1]

        if self._edge_count == 0:
            raise Exception("graph topo is not set")
        return self._edge_count

    @property
    def node_count(self):
        """Node count of graph"""
        if self._node_count > 0:
            return self._node_count
        if self._adj_csr is not None:
            self._node_count = self._adj_csr.indptr.shape[0]
        if self._adj_coo is not None:
            self._node_count = (np.unique(self.adj_coo[0])).shape[0]

        if self._node_count == 0:
            raise Exception("graph topo is not set")

        return self._node_count

    @property
    def is_batched(self) -> bool:
        return self.batch_meta is not None

    @property
    def batch_meta(self) -> BatchMeta:
        return self._batch_meta

    @batch_meta.setter
    def batch_meta(self, batch_meta: BatchMeta):
        self._batch_meta = batch_meta

    @node_count.setter
    def node_count(self, node_count):
        self._node_count = node_count

    @edge_count.setter
    def edge_count(self, edge_count):
        self._edge_count = edge_count

    def __getitem__(self, graph_idx) -> 'MindHomoGraph':
        if not self.is_batched:
            return self
        res = MindHomoGraph()
        node_count, edge_count = self.batch_meta[graph_idx]
        res.adj_coo = self.adj_coo[:, self.batch_meta.graph_edges[graph_idx]:
                                   self.batch_meta.graph_edges[graph_idx + 1]]
        res.adj_coo -= - self.batch_meta.graph_nodes[graph_idx]
        res.node_count = node_count
        res.edge_count = edge_count
        return res

    ############################
    # Inner Function
    ############################
    def _check_csr(self):
        assert self._adj_csr is not None or self._adj_coo is not None
        if self._adj_csr is not None:
            return
        self._adj_csr = coo_to_csr()
        return

    def _check_coo(self):
        assert self._adj_csr is not None or self._adj_coo is not None
        if self._adj_coo is not None:
            return
        self._adj_coo = csr_to_coo(self._adj_csr)
        return


class MindHeteroGraph:
    """
    HeteroGeneous Graph
    """

    def __init__(self):
        self._rel_graphs: Dict = {}

        # Information For Batch
        self._batch_meta = None

    def add_graph(self, rel_graph: MindRelationGraph):
        self._rel_graphs[rel_graph.relation_type] = rel_graph

    ########################################
    # Query Graphs With Lazy Computation
    #######################################
    def successors(self, relation_type, src_node):

        return self._rel_graphs[relation_type].successors(src_node)

    def predecessors(self, dst_node):
        pass

    def out_degree(self, relation_type, src_node):
        return self._rel_graphs[relation_type].out_degree(src_node)

    def out_degrees(self, relation_type, src_nodes):
        pass

    def in_degree(self, relation_type, dst_node):
        pass

    def in_degrees(self, relation_type, dst_nodes):
        pass

    def format(self, relation_type, out_format):
        pass

    def nodes(self, relation_type):
        return self._rel_graphs[relation_type].nodes

    def edges(self, relation_type):
        return self._rel_graphs[relation_type].edges

    def sample_succeessors(self, relation_type, src_node, neighbor_num):
        return self._rel_graphs[relation_type].sample_successors(src_node, neighbor_num)
