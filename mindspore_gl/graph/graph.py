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

CsrAdjNameTuple = namedtuple("csr_adj", ['indptr', 'indices'])


class CsrAdj(CsrAdjNameTuple):
    """
    Build the CSR matrix nametuple.

    Args:
        indptr (numpy.ndarry): indptr of CSR matrix.
        indices (numpy.ndarry): indices of CSR matrix.

    Raises:
        TypeError: If `indptr` or `indices` is not a numpy.ndarray.
        TypeError: If `indptr` or `indices` is not a one dimesion array.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore_gl.graph.graph import CsrAdj
        >>> indptr = np.array([0, 2, 3, 6])
        >>> indices = np.array([0, 2, 2, 0, 1, 2])
        >>> csr_adj = CsrAdj(indptr, indices)
        >>> print(csr_adj)
        CsrAdj(indptr=array([0, 2, 3, 6]), indices=array([0, 2, 2, 0, 1, 2]))
    """
    def __init__(self, indptr, indices):
        if not isinstance(indptr, np.ndarray):
            raise TypeError("'indptr' type must be numpy.ndarray, but get {}.".format(type(indptr)))
        if len(indptr.shape) >= 2 and indptr.shape[1] >= 2:
            raise TypeError("'indptr' shape must 1 dimesion, but get {}.".format(indptr.shape))
        if not isinstance(indices, np.ndarray):
            raise TypeError("'indices' type must be numpy.ndarray, but get {}.".format(type(indices)))
        if len(indices.shape) >= 2 and indices.shape[1] >= 2:
            raise TypeError("'indices' shape must 1 dimesion, but get {}.".format(indices.shape))
        super(CsrAdj, self).__init__()

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
        Set topology for relation graph by either csr_adj.

        Args:
            adj_csr(CsrAdj): csr format description of adjacent matrix.
            node_dict(dict, optional): edge ids for each edge.
            edge_ids(Unumpy.ndarray, optional): global->local node id.
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
            - bool, indicate if node is in this graph

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
        graph_nodes(numpy.array): array of accumulated node sum for graphs in batched graph (first element is 0).
        graph_edges(numpy.array): array of accumulated edge sum for graphs in batched graph (first element is 0).

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindspore_gl.graph import BatchMeta
        >>> graph_nodes = np.array([0, 20, 40, 60, 80])
        >>> graph_edges = np.array([0, 100, 200, 300, 400])
        >>> graph = BatchMeta(graph_nodes, graph_edges)
        >>> print(graph[1])
        (20, 100)

    """
    def __init__(self, graph_nodes, graph_edges):
        self._graph_nodes = graph_nodes
        self._graph_edges = graph_edges
        # For Lazy Computation
        self._node_map_idx = None
        self._edge_map_idx = None

    @property
    def graph_nodes(self):
        """
        Nodes array of graph.

        Returns:
            - numpy.array, accumulated node sum for graphs in batched graph(first element is 0)

        """
        return self._graph_nodes

    @property
    def graph_edges(self):
        """
        Edges array of graph.

        Returns:
            - numpy.array, accumulated edge sum for graphs in batched graph(first element is 0)
        """
        return self._graph_edges

    @property
    def graph_count(self):
        """
        Graph numbers.

        Returns:
            - int, total graph count in this batched graph

        """
        return self._graph_edges.shape[0] - 1

    @property
    def node_map_idx(self):
        """
        Index of node list.

        Returns:
            - numpy.array, array indicate graph index for each node

        """
        if self._node_map_idx is not None:
            return self._node_map_idx
        self._node_map_idx = np.zeros([self._graph_nodes[-1]], dtype=np.int32)
        kernel.set_node_map_idx(self._node_map_idx, self._graph_nodes)
        return self._node_map_idx

    @property
    def edge_map_idx(self):
        """
        Index of edge list.

        Returns:
            - numpy.array, array indicate graph index for each edge
        """
        if self._edge_map_idx is not None:
            return self._edge_map_idx
        self._edge_map_idx = np.zeros([self._graph_edges[-1]], dtype=np.int32)
        kernel.set_edge_map_idx(self._edge_map_idx, self._graph_edges)
        return self._edge_map_idx

    def __getitem__(self, graph_idx):
        """
        Return node count and edge count for idx graph

        Args:
            graph_idx(int): graph idx for query

        Returns:
            - (int, int), (node_count, edge_count)
        """
        assert graph_idx < self.graph_count, "index out of range"
        return (self.graph_nodes[graph_idx + 1] - self.graph_nodes[graph_idx],
                self.graph_edges[graph_idx + 1] - self.graph_edges[graph_idx])


class MindHomoGraph:
    """
    Build homo graph.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import networkx
        >>> from scipy.sparse import csr_matrix
        >>> from mindspore_gl.graph import MindHomoGraph, CsrAdj
        >>> node_count = 20
        >>> edge_prob = 0.1
        >>> graph = networkx.generators.random_graphs.fast_gnp_random_graph(node_count, edge_prob)
        >>> edge_array = np.transpose(np.array(list(graph.edges)))
        >>> row = edge_array[0]
        >>> col = edge_array[1]
        >>> data = np.ones(row.shape)
        >>> csr_mat = csr_matrix((data, (row, col)), shape=(node_count, node_count))
        >>> generated_graph = MindHomoGraph()
        >>> node_dict = {idx: idx for idx in range(node_count)}
        >>> edge_count = col.shape[0]
        >>> edge_ids = np.array(list(range(edge_count))).astype(np.int32)
        >>> generated_graph.set_topo(CsrAdj(csr_mat.indptr.astype(np.int32), csr_mat.indices.astype(np.int32)),
        ... node_dict, edge_ids)
        >>> print(generated_graph.neighbors(0))
        # results will be random for suffle
        [10 14]

    """
    def __init__(self):

        self._node_dict = None
        self._node_ids = None
        self._edge_ids = None

        self._adj_csr: CsrAdj = None
        self._adj_coo = None

        self._node_count = 0
        self._edge_count = 0

        "Batch Meta Info"
        self._batch_meta = None

    def set_topo(self, adj_csr: CsrAdj, node_dict, edge_ids: np.ndarray):
        """
        Initialize CSR Graph.

        Args:
            adj_csr(CsrAdj): adjacency matrix of graph, CSR type.
            node_dict(dict): node ID dict.
            edge_ids(numpy.ndarray): array of edges.
        """
        self._adj_csr = adj_csr
        self._node_dict = node_dict
        self._edge_ids = edge_ids
        self._node_ids = np.array(list(node_dict.keys()))

    def set_topo_coo(self, adj_coo, node_dict=None, edge_ids: np.ndarray = None):
        """
        Initialize COO Graph.

        Args:
            adj_coo(numpy.ndarray): adjacency matrix of graph, COO type.
            node_dict(dict, optional): node ID dict. Default: ``None``.
            edge_ids(numpy.ndarray, optional): array of edges. Default: ``None``.
        """
        self._adj_coo = adj_coo
        self._node_dict = node_dict
        self._node_ids = None if node_dict is None else np.array(list(node_dict.keys()))
        self._edge_ids = edge_ids

    def neighbors(self, node):
        """
        Query neighbors nodes.

        Args:
            node(int): node index.

        Returns:
            - numpy.ndarray, sampled node.
        """
        self._check_csr()
        mapped_idx = self._node_dict.get(node, None)
        assert mapped_idx is not None
        neighbor_start = self._adj_csr.indptr[mapped_idx]
        neighbor_end = self._adj_csr.indptr[mapped_idx + 1]
        neighbors = self._adj_csr.indices[neighbor_start: neighbor_end]
        node_ids = self._node_ids[neighbors]
        return node_ids

    def degree(self, node):
        """
        Query With node degree.

        Args:
            node(int): node index.

        Returns:
            - int, degree of node.
        """
        self._check_csr()
        mapped_idx = self._node_dict.get(node, None)
        assert mapped_idx is not None
        neighbor_start = self._adj_csr.indptr[mapped_idx]
        neighbor_end = self._adj_csr.indptr[mapped_idx + 1]
        return neighbor_end - neighbor_start

    @property
    def adj_csr(self):
        """
        CSR adj matrix.

        Returns:
            - mindspore_gl.graph.csr_adj, CSR graph.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> adj_csr = graph.adj_csr
        """
        self._check_csr()
        return self._adj_csr

    @property
    def adj_coo(self):
        """
        COO adj matrix.

        Returns:
            - numpy.ndarray, coo graph.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> adj_coo = graph.adj_coo
        """
        self._check_coo()
        return self._adj_coo

    @adj_coo.setter
    def adj_coo(self, adj_coo):
        del self._adj_csr
        self._adj_csr = None
        self._adj_coo = adj_coo

    @property
    def edge_count(self):
        """Edge count of graph.

        Returns:
            - int, edge numbers.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> edge_count = graph.edge_count
        """
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
        """Node count of graph.

        Returns:
            - int, nodes numbers.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> node_count = graph.node_count
        """
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
        """If the graph be batched.

        Returns:
            - bool, the graph be batched return True, else return False.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> is_batched = graph.is_batched
        """
        return self.batch_meta is not None

    @property
    def batch_meta(self) -> BatchMeta:
        """Batched graph meta info.

        Returns:
            - mindspore_gl.graph.BatchMeta, batched graph meta info.

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> batch_meta = graph.batch_meta
        """
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

    def _check_csr(self):
        """graph type check, is csr"""
        assert self._adj_csr is not None or self._adj_coo is not None
        if self._adj_csr is not None:
            return
        self._adj_csr = coo_to_csr()
        return

    def _check_coo(self):
        """graph type check, is coo"""
        assert self._adj_csr is not None or self._adj_coo is not None
        if self._adj_coo is not None:
            return
        self._adj_coo = csr_to_coo(self._adj_csr)
        return


class MindHeteroGraph:
    """
    HeteroGeneous Graph.
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
