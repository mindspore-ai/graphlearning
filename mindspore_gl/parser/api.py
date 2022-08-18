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
"""API"""
import mindspore as ms


class SrcVertex:
    """Source Vertex"""

    def __init__(self):
        raise NotImplementedError("SrcVertex Init: not implemented!")


class DstVertex:
    """Destination Vertex"""

    def __init__(self, innb):
        self._innbs = [innb]
        self.in_edges = None

    @property
    def innbs(self):
        """
        Return a list of src_vertex of current vertex.

        Examples:
            >>> for v in g.dst_vertex:
            ...     v.h = g.sum([u.h for u in v.innbs])
        """

        return self._innbs

    @property
    def inedges(self):
        """
        Return a list of (src, edge) tuples for current vertex.

        Examples:
            >>> for v in g.dst_vertex:
            ...     [u.a + e.b for u,e in v.inedges]
        """

        assert self.in_edges is not None
        return self.in_edges


class Edge:
    """Edge"""

    def __init__(self, src, dst):
        dst.in_edges = [(src, self)]
        self._src = src
        self._dst = dst


class Graph:
    """
    Graph class.

    This is the class which should be annotated in \
        construct function for GNNCell class.
    """

    def __init__(self):
        self._src_vertex = SrcVertex()
        self._dst_vertex = [DstVertex(self._src_vertex)]
        self._edge = [Edge(self._src_vertex, dst_v)
                      for dst_v in self._dst_vertex]

    @property
    def dst_vertex(self):
        """
        Return a list of destination vertex that only supports\
             iterate its innbs.

        Examples:
            >>> for v in g.dst_vertex:
            ...     pass
        """
        return self._dst_vertex

    @property
    def src_vertex(self):
        """
        Return a list of vertex that only supports iterate with its outnbs

        Examples:
            >>> for u in g.src_vertex:
            ...     pass
        """
        return self._src_vertex

    @property
    def src_idx(self):
        r"""
        A tensor with shape :math:`(N\_EDGES)`, represents the source node \
            index of COO edge matrix.
        """

    @property
    def dst_idx(self):
        r"""
        A tensor with shape :math:`(N\_EDGES)`, represents the destination \
             node index of COO edge matrix.
        """

    @property
    def n_nodes(self):
        """
        An integer, represent the nodes count of the graph.
        """

    @property
    def n_edges(self):
        """
        An integer, represent the edges count of the graph.
        """

    def set_vertex_attr(self, feat_dict):
        r"""
        Set attributes for vertices in vertex-centric environment.
        Keys will be attribute's name, values will be attributes' data.

        Note:
            set_vertex_attr is equals to set_src_attr + set_dst_attr.

        Args:
            feat_dict (Dict): key type: str, value type: recommend tensor of \
                shape :math:`(N\_NODES, F)`, :math:`F` is the dimension of the node feature.

        Raises:
            TypeError: If `feat_dict` is not a Dict.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gl import Graph, GraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> n_nodes = 9
            >>> n_edges = 11
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
            >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
            >>> node_feat = ms.Tensor([[1], [2], [1], [2], [0], [1], [2], [3], [1]], ms.float32)
            ...
            >>> class TestSetVertexAttr(GNNCell):
            ...     def construct(self, x, g: Graph):
            ...         g.set_vertex_attr({"h": x})
            ...         return [v.h for v in g.dst_vertex] * [u.h for u in g.src_vertex]
            ...
            >>> ret = TestSetVertexAttr()(node_feat, *graph_field.get_graph()).asnumpy().tolist()
            >>> print(ret)
                [[1.0], [4.0], [1.0], [4.0], [0.0], [1.0], [4.0], [9.0], [1.0]]
        """

    def set_src_attr(self, feat_dict):
        r"""
        Set attributes for source vertices in vertex-centric environment.
        Keys will be attribute's name, values will be attributes' data.

        Args:
            feat_dict (Dict): key type: str, value type: recommend tensor of
                shape :math:`(N\_NODES, F)`, :math:`F` is the dimension of the node feature.

        Raises:
            TypeError: If `feat_dict` is not a Dict.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gl import Graph, GraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> n_nodes = 9
            >>> n_edges = 11
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
            >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
            >>> node_feat = ms.Tensor([[1], [2], [1], [2], [0], [1], [2], [3], [1]], ms.float32)
            ...
            >>> class TestSetSrcAttr(GNNCell):
            ...     def construct(self, x, g: Graph):
            ...         g.set_src_attr({"h": x})
            ...         return [u.h for u in g.src_vertex]
            ...
            >>> ret = TestSetSrcAttr()(node_feat, *graph_field.get_graph()).asnumpy().tolist()
            >>> print(ret)
                [[1.0], [2.0], [1.0], [2.0], [0.0], [1.0], [2.0], [3.0], [1.0]]
        """

    def set_dst_attr(self, feat_dict):
        r"""
        Set attributes for destination vetices in vertex-centric environment
        Keys will be attribute's name, values will be attributes' data.

        Args:
            feat_dict (Dict): key type: str, value type: recommend tensor of
                shape :math:`(N\_NODES, F)`, :math:`F` is the dimension of the node feature.

        Raises:
            TypeError: If `feat_dict` is not a Dict.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gl import Graph, GraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> n_nodes = 9
            >>> n_edges = 11
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
            >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
            >>> node_feat = ms.Tensor([[1], [2], [1], [2], [0], [1], [2], [3], [1]], ms.float32)
            ...
            >>> class TestSetDstAttr(GNNCell):
            ...     def construct(self, x, g: Graph):
            ...         g.set_dst_attr({"h": x})
            ...         return [v.h for v in g.dst_vertex]
            ...
            >>> ret = TestSetDstAttr()(node_feat, *graph_field.get_graph()).asnumpy().tolist()
            >>> print(ret)
                [[1.0], [2.0], [1.0], [2.0], [0.0], [1.0], [2.0], [3.0], [1.0]]
        """

    def set_edge_attr(self, feat_dict):
        r"""
        Set attributes for edges in vertex-centric environment.
        Keys will be attribute's name, values will be attributes' data.

        Args:
            feat_dict (Dict): key type: str, value type: recommend feature tensor
                of shape :math:`(N\_EDGES, *)`, :math:`*` is the shape of the feature per edge.
                Recommend the shape of value is :math:`(N\_EDGES, 1)` when the feature dimension is 1.

        Raises:
            TypeError: If `feat_dict` is not a Dict.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gl import Graph, GraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> n_nodes = 9
            >>> n_edges = 11
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
            >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
            >>> node_feat = ms.Tensor([[1], [2], [1], [2], [0], [1], [2], [3], [1]], ms.float32)
            >>> edge_feat = ms.Tensor([[1], [2], [1], [3], [1], [4], [1], [5], [1], [1], [1]], ms.float32)
            ...
            >>> class TestSetEdgeAttr(GNNCell):
            ...     def construct(self, nh, eh, g: Graph):
            ...         g.set_vertex_attr({"nh": nh})
            ...         g.set_edge_attr({"eh": eh})
            ...         for v in g.dst_vertex:
            ...             v.h = g.sum([u.nh * e.eh for u, e in v.inedges])
            ...         return [v.h for v in g.dst_vertex]
            ...
            >>> ret = TestSetEdgeAttr()(node_feat, edge_feat, *graph_field.get_graph()).asnumpy().tolist()
            >>> print(ret)
                [[2.0], [2.0], [0.0], [0.0], [14.0], [6.0], [1.0], [0.0], [3.0]]
        """

    def set_graph_attr(self, feat_dict):
        """
        Set attributes for the whole graph in vertex-centric environment.
        Keys will be attribute's name, values will be attributes' data.

        Args:
            feat_dict (Dict): key type: str, value type: recommend feature tensor
                for the whole graph.

        Raises:
            TypeError: If `feat_dict` is not a Dict.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gl import Graph, GraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> n_nodes = 9
            >>> n_edges = 11
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
            >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
            >>> g_attr = ms.Tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], ms.float32)
            >>> v_attr = ms.Tensor([1.0, 1.0], ms.float32)
            ...
            >>> class TestSetGraphAttr(GNNCell):
            ...     def construct(self, vh, gh, g: Graph):
            ...         g.set_graph_attr({"x": gh})
            ...         g.set_vertex_attr({"h": vh})
            ...         for v in g.dst_vertex:
            ...             v.h = g.sum([u.h * g.x for u in v.innbs])
            ...         return [v.h for v in g.dst_vertex]
            ...
            >>> ret = TestSetGraphAttr()(v_attr, g_attr, *graph_field.get_graph()).asnumpy().tolist()
            >>> print(ret)
                [[0.0, 1.0], [0.0, 2.0], [0.0, 0.0], [0.0, 0.0],
                 [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        """

    def sum(self, neigh_feat):
        r"""
        Aggregating node features from their neighbour and generates a
        node-level representation by aggregate function 'sum'.

        Args:
            neigh_feat (List[`SrcVertex` feature or `Edge` feature]): a list of `SrcVertex` or `Edge` attribute
                represents the neighbour nodes or edges feature, with shape :math:`(N, F)`,
                :math:`N` is the number of `SrcVertex` or `Edge`,
                :math:`F` is the feature dimension of the `SrcVertex` or `Edge` attribute.

        Returns:
            Tensor, a tensor with shape :math:`(N, F)`, :math:`N` is the number of nodes of the graph,
            :math:`F` is the feature dimension of the node.

        Raises:
            TypeError: If `neigh_feat` is not a list of `Edge` or `SrcVertex`.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gl import Graph, GraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> n_nodes = 9
            >>> n_edges = 11
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
            >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
            >>> node_feat = ms.Tensor([[1], [2], [1], [2], [0], [1], [2], [3], [1]], ms.float32)
            ...
            >>> class TestSum(GNNCell):
            ...     def construct(self, x, g: Graph):
            ...         g.set_vertex_attr({"x": x})
            ...         for v in g.dst_vertex:
            ...             v.h = g.sum([u.x for u in v.innbs])
            ...         return [v.h for v in g.dst_vertex]
            ...
            >>> ret = TestSum()(node_feat, *graph_field.get_graph()).asnumpy().tolist()
            >>> print(ret)
                [[1.0], [2.0], [0.0], [0.0], [3.0], [2.0], [1.0], [0.0], [3.0]]
        """

    def max(self, neigh_feat):
        r"""
        Aggregating node features from their neighbour and generates
        a node-level representation by aggregate function 'max'.

        Args:
            neigh_feat (List[`SrcVertex` feature or `Edge` feature]): a list of `SrcVertex` or `Edge` attributes
                represents the neighbour nodes or edges feature, with shape :math:`(N, F)`,
                :math:`N` is the number of `SrcVertex` or `Edge`,
                :math:`F` is the feature dimension of the `SrcVertex` or `Edge` attribute.

        Returns:
            Tensor, a tensor with shape :math:`(N, F)`, :math:`N` is the number of nodes of the graph,
            :math:`F` is the feature dimension of the node.

        Raises:
            TypeError: If `neigh_feat` is not a list of `Edge` or `SrcVertex`.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gl import Graph, GraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> n_nodes = 9
            >>> n_edges = 11
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
            >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
            >>> node_feat = ms.Tensor([[1], [2], [1], [2], [0], [1], [2], [3], [1]], ms.float32)
            ...
            >>> class TestMax(GNNCell):
            ...     def construct(self, x, g: Graph):
            ...         g.set_vertex_attr({"x": x})
            ...         for v in g.dst_vertex:
            ...             v.h = g.max([u.x for u in v.innbs])
            ...         return [v.h for v in g.dst_vertex]
            ...
            >>> ret = TestMax()(node_feat, *graph_field.get_graph()).asnumpy().tolist()
            >>> print(ret)
                [[1.0], [1.0], [0.0], [0.0], [2.0], [2.0], [1.0], [0.0], [1.0]]
        """

    def min(self, neigh_feat):
        r"""
        Aggregating node features from their neighbour and generates
        a node-level representation by aggregate function 'min'.

        Args:
            neigh_feat (List[`SrcVertex` feature or `Edge` feature]): a list of `SrcVertex` or `Edge` attributes
                represents the neighbour nodes or edges feature, with shape :math:`(N, F)`,
                :math:`N` is the number of `SrcVertex` or `Edge`,
                :math:`F` is the feature dimension of the `SrcVertex` or `Edge` attribute.

        Returns:
            Tensor, a tensor with shape :math:`(N, F)`, :math:`N` is the number of nodes of the graph,
            :math:`F` is the feature dimension of the node.

        Raises:
            TypeError: If `neigh_feat` is not a list of `Edge` or `SrcVertex`.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gl import Graph, GraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> n_nodes = 9
            >>> n_edges = 11
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
            >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
            >>> node_feat = ms.Tensor([[1], [2], [1], [2], [0], [1], [2], [3], [1]], ms.float32)
            ...
            >>> class TestMin(GNNCell):
            ...     def construct(self, x, g: Graph):
            ...         g.set_vertex_attr({"x": x})
            ...         for v in g.dst_vertex:
            ...             v.h = g.min([u.x for u in v.innbs])
            ...         return [v.h for v in g.dst_vertex]
            ...
            >>> ret = TestMin()(node_feat, *graph_field.get_graph()).asnumpy().tolist()
            >>> print(ret)
                [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
        """

    def avg(self, neigh_feat):
        r"""
        Aggregating node features from their neighbour and generates a
        node-level representation by aggregate function 'avg'.

        Args:
            neigh_feat (List[`SrcVertex` feature or `Edge` feature]): a list of `SrcVertex` or `Edge` attributes
                represents the neighbour nodes or edges feature, with shape :math:`(N, F)`,
                :math:`N` is the number of `SrcVertex` or `Edge`,
                :math:`F` is the feature dimension of the `SrcVertex` or `Edge` attribute.

        Returns:
            Tensor, a tensor with shape :math:`(N, F)`, :math:`N` is the number of nodes of the graph,
            :math:`F` is the feature dimension of the node.

        Raises:
            TypeError: If `neigh_feat` is not a list of `Edge` or `SrcVertex`.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import math
            >>> import mindspore as ms
            >>> from mindspore_gl import Graph, GraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> n_nodes = 9
            >>> n_edges = 11
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
            >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
            >>> node_feat = ms.Tensor([[1], [2], [1], [2], [0], [1], [2], [3], [1]], ms.float32)
            ...
            >>> class TestAvg(GNNCell):
            ...     def construct(self, x, g: Graph):
            ...         g.set_vertex_attr({"x": x})
            ...         for v in g.dst_vertex:
            ...             v.h = g.avg([u.x for u in v.innbs])
            ...         return [v.h for v in g.dst_vertex]
            ...
            >>> ret = TestAvg()(node_feat, *graph_field.get_graph()).asnumpy().tolist()
            >>> NAN = 1e9
            >>> for row in ret:
            ...     if math.isnan(row[0]):
            ...         row[0] = NAN
            >>> print(ret)
                [[1.0], [1.0], [1000000000.0], [0.0], [1.5], [2.0], [1.0], [1000000000.0], [1.0]]
        """

    def dot(self, feat_x, feat_y):
        r"""
        Dot mul operation for two node Tensors.

        Args:
            feat_x (`SrcVertex` feature or `DstVertex` feature): the arttribute of `SrcVertex` or `DstVertex`
                represent feature tensor of graph nodes with shape :math:`(N, F)`,
                :math:`N` is the number of nodes of the graph,
                :math:`F` is the feature dimension of the node.
            feat_y (`SrcVertex` feature or `DstVertex` feature): the arttribute of `SrcVertex` or `DstVertex`
                represent feature tensor of graph nodes with shape :math:`(N, F)`,
                :math:`N` is the number of nodes of the graph,
                :math:`F` is the feature dimension of the node.

        Returns:
            Tensor, a tensor with shape :math:`(N, 1)`, :math:`N` is the number of nodes of the graph.

        Raises:
            TypeError: If `feat_x` is not in the 'mul' operation support types [Tensor,Number,List,Tuple].
            TypeError: If `feat_y` is not in the 'mul' operation support types [Tensor,Number,List,Tuple].

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gl import Graph, GraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> n_nodes = 9
            >>> n_edges = 11
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
            >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
            >>> node_feat = ms.Tensor([[1], [2], [1], [2], [0], [1], [2], [3], [1]], ms.float32)
            ...
            >>> class TestDot(GNNCell):
            ...     def construct(self, x, g: Graph):
            ...         g.set_vertex_attr({"src": x, "dst": x})
            ...         for v in g.dst_vertex:
            ...             v.h = [g.dot(v.src, u.dst) for u in v.innbs]
            ...         return [v.h for v in g.dst_vertex]
            ...
            >>> ret = TestDot()(node_feat, *graph_field.get_graph()).asnumpy().tolist()
            >>> print(ret)
                [[2.0], [1.0], [2.0], [2.0], [0.0], [0.0], [2.0], [0.0], [1.0], [1.0], [1.0]]
        """

    def topk_nodes(self, node_feat, k, sortby=None):
        r"""
        Return a graph-level representation by a graph-wise top-k
        on node features.

        If sortby is set to None, the function would perform top-k
        on all dimensions independently.

        Args:
            node_feat (Tensor): A tensor represent the node feature,
                with shape :math:`(N\_NODES, F)`. :math:`F` is the dimension of the node feature.
            k (int): Represent how many nodes for top-k.
            sortby (int): Sort according to which feature. If is None,
                all features are sorted independently.  Default is None.

        Note:
            The value participated in the sort by axis (all value if sortby is
            None) should be greater than zero.
            Due to the reason that we create zero value for padding
            and they may cover the features.

        Returns:
            - **topk_output** (Tensor) - a tensor with shape :math:`(B, K, F)`,
              where :math:`B` is the batch size of the input graph.
              :math:`K` is the input 'k', :math:`F` is the feature size.
            - **topk_indices** (Tensor), - a tensor with shape
              :math:`(B, K)`(:math:`(B, K, F)` if sortby is set to None),
              where :math:`B` is the batch size of the input graph,
              :math:`F` is the feature size.

        Raises:
            TypeError: If `node_feat` is not a Tensor.
            TypeError: If `k` is not an int.
            ValueError: If `sortby` is not an int.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gl import BatchedGraph, BatchedGraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> node_feat = ms.Tensor([
            ...     [1, 2, 3, 4],
            ...     [2, 4, 1, 3],
            ...     [1, 3, 2, 4],
            ...     [9, 7, 5, 8],
            ...     [8, 7, 6, 5],
            ...     [8, 6, 4, 6],
            ...     [1, 2, 1, 1],
            ... ], ms.float32)
            ...
            >>> n_nodes = 7
            >>> n_edges = 8
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
            >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
            ...
            >>> class TestTopkNodes(GNNCell):
            ...     def construct(self, x, g: Graph):
            ...         return g.topk_nodes(x, 2, 1)
            ...
            >>> output, indices = TestTopkNodes()(node_feat, *graph_field.get_graph())
            >>> output = output.asnumpy().tolist()
            >>> indices = indices.asnumpy().tolist()
            >>> print(output)
                [[9.0, 7.0, 5.0, 8.0], [8.0, 7.0, 6.0, 5.0]]
            >>> print(indices)
                [3, 4]
        """

    def topk_edges(self, node_feat, k, sortby=None):
        r"""
        Return a graph-level representation by a graph-wise top-k
        on node features.

        If sortby is set to None, the function would perform top-k
        on all dimensions independently.

        Args:
            node_feat (Tensor): A tensor represent the node feature,
                with shape :math:`(N\_NODES, F)`. :math:`F` is the dimension of the node feature.
            k (int): Represent how many nodes for top-k.
            sortby (int): Sort according to which feature. If is None,
                all features are sorted independently.  Default is None.

        Note:
            The value participated in the sort by axis (all value if sortby is
            None) should be greater than zero.
            Due to the reason that we create zero value for padding
            and they may cover the features.

        Returns:
            - **topk_output** (Tensor) - a tensor with shape :math:`(B, K, F)`,
              where :math:`B` is the batch size of the input graph.
              :math:`K` is the input 'k', :math:`F` is the feature size.
            - **topk_indices** (Tensor), - a tensor with shape
              :math:`(B, K)`(:math:`(B, K, F)` if sortby is set to None),
              where :math:`B` is the batch size of the input graph,
              :math:`F` is the feature size.

        Raises:
            TypeError: If `node_feat` is not a Tensor.
            TypeError: If `k` is not an int.
            ValueError: If `sortby` is not an int.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gl import BatchedGraph, BatchedGraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> node_feat = ms.Tensor([
            ...     [1, 2, 3, 4],
            ...     [2, 4, 1, 3],
            ...     [1, 3, 2, 4],
            ...     [9, 7, 5, 8],
            ...     [8, 7, 6, 5],
            ...     [8, 6, 4, 6],
            ...     [1, 2, 1, 1],
            ... ], ms.float32)
            ...
            >>> n_nodes = 7
            >>> n_edges = 8
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
            >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
            ...
            >>> class TestTopkEdges(GNNCell):
            ...     def construct(self, x, g: Graph):
            ...         return g.topk_edges(x, 2, 1)
            ...
            >>> output, indices = TestTopkEdges()(node_feat, *graph_field.get_graph())
            >>> output = output.asnumpy().tolist()
            >>> indices = indices.asnumpy().tolist()
            >>> print(output)
                [[9.0, 7.0, 5.0, 8.0], [8.0, 7.0, 6.0, 5.0]]
            >>> print(indices)
                [3, 4]
        """

    def in_degree(self):
        r"""
        Get the in degree of each node in a graph.

        Returns:
            Tensor, a tensor with shape :math:`(N, 1)`,
            represent the in degree of each node, :math:`N` is the number of nodes of the graph.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gl import Graph, GraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> n_nodes = 9
            >>> n_edges = 11
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
            >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
            ...
            >>> class TestInDegree(GNNCell):
            ...     def construct(self, g: Graph):
            ...         return g.in_degree()
            ...
            >>> ret = TestInDegree()(*graph_field.get_graph()).asnumpy().tolist()
            >>> print(ret)
                [[1], [2], [0], [1], [2], [1], [1], [0], [3]]
        """

    def out_degree(self):
        r"""
        Get the out degree of each node in a graph.

        Returns:
            Tensor, a tensor with shape :math:`(N, 1)`,
            represent the out degree of each node, :math:`N` is the number of nodes of the graph.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gl import Graph, GraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> n_nodes = 9
            >>> n_edges = 11
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
            >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
            ...
            >>> class TestOutDegree(GNNCell):
            ...     def construct(self, g: Graph):
            ...         return g.out_degree()
            ...
            >>> ret = TestOutDegree()(*graph_field.get_graph()).asnumpy().tolist()
            >>> print(ret)
                [[1], [0], [2], [1], [1], [2], [1], [0], [3]]
        """

    def adj_to_dense(self):
        r"""
        Get the dense adjacent matrix of the graph.

        Note:
            You must set vertex attr first due to the current
            limits of our system.

        Returns:
            Tensor, a tensor with shape :math:`(N, N)`, :math:`N` is the number of nodes of the graph.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gl import Graph, GraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> n_nodes = 9
            >>> n_edges = 11
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
            >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
            ...
            >>> class TestAdjToDense(GNNCell):
            ...     def construct(self, g: Graph):
            ...         return g.adj_to_dense()
            ...
            >>> ret = TestAdjToDense()(*graph_field.get_graph()).asnumpy().tolist()
            >>> print(ret)
                [[0, 1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 1, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 3]]
        """


class BatchedGraph(Graph):
    """
    Batched Graph class.

    This is the class which should be annotated in
    construct function for GNNCell class.
    """

    @property
    def ver_subgraph_idx(self):
        r"""
        A tensor with shape :math:`(N)`, indicates each node belonging
        to which subgraph, :math:`N` is the number of the nodes of the graph.
        """

    @property
    def edge_subgraph_idx(self):
        r"""
        A tensor with shape :math:`(N\_EDGES,)`, indicates each edge belonging to which subgraph.
        """

    @property
    def graph_mask(self):
        r"""
        A tensor with shape :math:`(N\_GRAPHS,)`, indicates whether the subgraph is exist.
        """

    @property
    def n_graphs(self):
        """
        An integer, represent the graphs count of the batched graph.
        """

    def node_mask(self):
        r"""
        Get the node mask after padding.

        The node mask is calculated according to the graph_mask and
        ver_subgraph_idx.

        Returns:
            Tensor, a tensor with shape :math:`(N\_NODES, )`. Inside tensor, 1 represent the node exists and
            0 represent the node is generated by padding.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gl import BatchedGraph, BatchedGraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> n_nodes = 9
            >>> n_edges = 11
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
            >>> ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 2, 2], ms.int32)
            >>> edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2], ms.int32)
            >>> graph_mask = ms.Tensor([1, 1, 0], ms.int32)
            >>> batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges,
            ...                                         ver_subgraph_idx, edge_subgraph_idx, graph_mask)
            ...
            >>> class TestNodeMask(GNNCell):
            ...     def construct(self, bg: BatchedGraph):
            ...         return bg.node_mask()
            ...
            >>> ret = TestNodeMask()(*batched_graph_field.get_batched_graph()).asnumpy().tolist()
            >>> print(ret)
                [1, 1, 1, 1, 1, 1, 1, 0, 0]
        """

    def edge_mask(self):
        r"""
        Get the edge mask after padding.

        The edge mask is calculated according to the graph_mask and
        ver_subgraph_idx.

        Returns:
            Tensor, a tensor with shape :math:`(N\_EDGES,)`.
            Inside tensor, 1 represent the edge exists and 0 represent the edge is generated by padding.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gl import BatchedGraph, BatchedGraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> n_nodes = 9
            >>> n_edges = 11
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
            >>> ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 2, 2], ms.int32)
            >>> edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2], ms.int32)
            >>> graph_mask = ms.Tensor([1, 1, 0], ms.int32)
            >>> batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges,
            ...                                         ver_subgraph_idx, edge_subgraph_idx, graph_mask)
            ...
            >>> class TestEdgeMask(GNNCell):
            ...     def construct(self, bg: BatchedGraph):
            ...         return bg.edge_mask()
            ...
            >>> ret = TestEdgeMask()(*batched_graph_field.get_batched_graph()).asnumpy().tolist()
            >>> print(ret)
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
        """

    def num_of_nodes(self):
        r"""
        Get the number of nodes of each subgraph in a batched graph.

        Returns:
            Tensor, a tensor with shape :math:`(N\_GRAPHS, 1)` represent each subgraph contains how many nodes.

        Note:
            After padding operation, a not existing subgraph is created
            and all not existing nodes created belong to this subgraph.
            If you want to clear it, you need to multiply it
            with a graph mask manually.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gl import BatchedGraph, BatchedGraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> n_nodes = 9
            >>> n_edges = 11
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
            >>> ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 2, 2], ms.int32)
            >>> edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2], ms.int32)
            >>> graph_mask = ms.Tensor([1, 1, 0], ms.int32)
            >>> batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges,
            ...                                         ver_subgraph_idx, edge_subgraph_idx, graph_mask)
            ...
            >>> class TestNumOfNodes(GNNCell):
            ...     def construct(self, bg: BatchedGraph):
            ...         return bg.num_of_nodes()
            ...
            >>> ret = TestNumOfNodes()(*batched_graph_field.get_batched_graph()).asnumpy().tolist()
            >>> print(ret)
                [[3], [4], [2]]
        """

    def num_of_edges(self):
        r"""
        Get the number of edges of each subgraph in a batched graph.

        Returns:
            Tensor, a tensor with shape :math:`(N\_GRAPHS, 1)`,
            represent each subgraph contains how many edges.

        Note:
            After padding operation, a not existing subgraph is created
            and all not existing edges created belong to this subgraph.
            If you want to clear it, you need to multiply it
            with a graph mask manually.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gl import BatchedGraph, BatchedGraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> n_nodes = 9
            >>> n_edges = 11
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
            >>> ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 2, 2], ms.int32)
            >>> edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2], ms.int32)
            >>> graph_mask = ms.Tensor([1, 1, 0], ms.int32)
            >>> batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges,
            ...                                         ver_subgraph_idx, edge_subgraph_idx, graph_mask)
            ...
            >>> class TestNumOfEdges(GNNCell):
            ...     def construct(self, bg: BatchedGraph):
            ...         return bg.num_of_edges()
            ...
            >>> ret = TestNumOfEdges()(*batched_graph_field.get_batched_graph()).asnumpy().tolist()
            >>> print(ret)
                [[3], [5], [3]]
        """

    def sum_nodes(self, node_feat):
        r"""
        Aggregating node features and generates a graph-level representation
        by aggregation type 'sum'.

        The node_feat should have shape :math:`(N\_NODES, F)`,
        Sum_nodes operation will aggregate the nodes
        feat according to ver_subgraph_idx.
        The output tensor will have a shape :math:`(N\_GRAPHS, F)`.
        :math:`F` is the dimension of the node feature.

        Args:
            node_feat (Tensor): a tensor represents the node feature,
                with shape :math:`(N\_NODES, F)`, :math:`F` is the dimension of the node node feature.

        Returns:
            Tensor, a tensor with shape :math:`(N\_GRAPHS, F)`, :math:`F` is the dimension of the node feature.

        Raises:
            TypeError: If `node_feat` is not a Tensor which is the type of operation 'shape'.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gl import BatchedGraph, BatchedGraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> node_feat = ms.Tensor([
            ...     # graph 1:
            ...     [1, 2, 3, 4],
            ...     [2, 4, 1, 3],
            ...     [1, 3, 2, 4],
            ...     # graph 2:
            ...     [9, 7, 5, 8],
            ...     [8, 7, 6, 5],
            ...     [8, 6, 4, 6],
            ...     [1, 2, 1, 1],
            ... ], ms.float32)
            ...
            >>> n_nodes = 7
            >>> n_edges = 8
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
            >>> ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
            >>> edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
            >>> graph_mask = ms.Tensor([1, 1], ms.int32)
            >>> batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges,
            ...                                         ver_subgraph_idx, edge_subgraph_idx, graph_mask)
            ...
            >>> class TestSumNodes(GNNCell):
            ...     def construct(self, x, bg: BatchedGraph):
            ...         return bg.sum_nodes(x)
            ...
            >>> ret = TestSumNodes()(node_feat, *batched_graph_field.get_batched_graph()).asnumpy().tolist()
            >>> print(ret)
                [[4.0, 9.0, 6.0, 11.0], [26.0, 22.0, 16.0, 20.0]]
        """

    def sum_edges(self, edge_feat):
        r"""
        Aggregating edge features and generates a graph-level representation
        by aggregation type 'sum'.

        The edge_feat should have shape :math:`(N\_EDGES, F)`.
        Sum_edges operation will aggregate the edge_feat.
        according to edge_subgraph_idx.
        The output tensor will have a shape :math:`(N\_GRAPHS, F)`.
        :math:`F` is the dimension of the edge feature.

        Args:
            edge_feat (Tensor): a tensor represents the edge feature,
                with shape :math:`(N\_EDGES, F)`. :math:`F` is the dimension of the edge attribute.

        Returns:
            Tensor, a tensor with shape :math:`(N\_GRAPHS, F)`. :math:`F` is the dimension of the edge attribute.

        Raises:
            TypeError: If `edge_feat` is not a Tensor which is the type of operation 'shape'.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gl import BatchedGraph, BatchedGraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> edge_feat = ms.Tensor([
            ...     # graph 1:
            ...     [1, 2, 3, 4],
            ...     [2, 4, 1, 3],
            ...     [1, 3, 2, 4],
            ...     # graph 2:
            ...     [9, 7, 5, 8],
            ...     [8, 7, 6, 5],
            ...     [8, 6, 4, 6],
            ...     [1, 2, 1, 1],
            ...     [3, 2, 3, 3],
            ... ], ms.float32)
            ...
            >>> n_nodes = 7
            >>> n_edges = 8
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
            >>> ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
            >>> edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
            >>> graph_mask = ms.Tensor([1, 1], ms.int32)
            >>> batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges,
            ...                                         ver_subgraph_idx, edge_subgraph_idx, graph_mask)
            ...
            >>> class TestSumEdges(GNNCell):
            ...     def construct(self, x, bg: BatchedGraph):
            ...         return bg.sum_edges(x)
            ...
            >>> ret = TestSumEdges()(edge_feat, *batched_graph_field.get_batched_graph()).asnumpy().tolist()
            >>> print(ret)
                [[4.0, 9.0, 6.0, 11.0], [29.0, 24.0, 19.0, 23.0]]
        """

    def max_nodes(self, node_feat):
        r"""
        Aggregating node features and generates a graph-level
        representation by aggregation type 'max'.

        The node_feat should have shape :math:`(N\_NODES, F)`.
        Max_nodes operation will aggregate the node_feat
        according to ver_subgraph_idx.
        The output tensor will have a shape :math:`(N\_GRAPHS, F)`.
        :math:`F` is the dimension of the node feature.

        Args:
            node_feat (Tensor): a tensor represents the node feature,
                with shape :math:`(N\_NODES, F)`. :math:`F` is the dimension of the node feature.

        Returns:
            Tensor, a tensor with shape :math:`(N\_GRAPHS, F)`, :math:`F` is the dimension of the node feature.

        Raises:
            TypeError: If `node_feat` is not a Tensor which is the type of operation 'shape'.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gl import BatchedGraph, BatchedGraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> node_feat = ms.Tensor([
            ...     # graph 1:
            ...     [1, 2, 3, 4],
            ...     [2, 4, 1, 3],
            ...     [1, 3, 2, 4],
            ...     # graph 2:
            ...     [9, 7, 5, 8],
            ...     [8, 7, 6, 5],
            ...     [8, 6, 4, 6],
            ...     [1, 2, 1, 1],
            ... ], ms.float32)
            ...
            >>> n_nodes = 7
            >>> n_edges = 8
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
            >>> ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
            >>> edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
            >>> graph_mask = ms.Tensor([1, 1], ms.int32)
            >>> batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges,
            ...                                         ver_subgraph_idx, edge_subgraph_idx, graph_mask)
            ...
            >>> class TestMaxNodes(GNNCell):
            ...     def construct(self, x, bg: BatchedGraph):
            ...         return bg.max_nodes(x)
            ...
            >>> ret = TestMaxNodes()(node_feat, *batched_graph_field.get_batched_graph()).asnumpy().tolist()
            >>> print(ret)
                [[2.0, 4.0, 3.0, 4.0], [9.0, 7.0, 6.0, 8.0]]
        """

    def max_edges(self, edge_feat):
        r"""
        Aggregating edge features and generates a graph-level
        representation by aggregation type 'max'.

        The edge_feat should have shape :math:`(N\_EDGES, F)`.
        Max_edges operation will aggregate the edge_feat
        according to edge_subgraph_idx.
        The output tensor will have a shape :math:`(N\_GRAPHS, F)`.
        :math:`F` is the dimension of the edge feature.

        Args:
            edge_feat (Tensor): a tensor represents the edge feature,
                with shape :math:`(N\_EDGES, F)`. :math:`F` is the dimension of the edge feature.

        Returns:
            Tensor, a tensor with shape :math:`(N\_GRAPHS, F)`. :math:`F` is the dimension of the edge feature.

        Raises:
            TypeError: If `edge_feat` is not a Tensor which is the type of operation 'shape'.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gl import BatchedGraph, BatchedGraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> edge_feat = ms.Tensor([
            ...     # graph 1:
            ...     [1, 2, 3, 4],
            ...     [2, 4, 1, 3],
            ...     [1, 3, 2, 4],
            ...     # graph 2:
            ...     [9, 7, 5, 8],
            ...     [8, 7, 6, 5],
            ...     [8, 6, 4, 6],
            ...     [1, 2, 1, 1],
            ...     [3, 2, 3, 3],
            ... ], ms.float32)
            ...
            >>> n_nodes = 7
            >>> n_edges = 8
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
            >>> ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
            >>> edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
            >>> graph_mask = ms.Tensor([1, 1], ms.int32)
            >>> batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges,
            ...                                         ver_subgraph_idx, edge_subgraph_idx, graph_mask)
            ...
            >>> class TestMaxEdges(GNNCell):
            ...     def construct(self, x, bg: BatchedGraph):
            ...         return bg.max_edges(x)
            ...
            >>> ret = TestMaxEdges()(edge_feat, *batched_graph_field.get_batched_graph()).asnumpy().tolist()
            >>> print(ret)
                [[2.0, 4.0, 3.0, 4.0], [9.0, 7.0, 6.0, 8.0]]
        """

    def avg_nodes(self, node_feat):
        r"""
        Aggregating node features and generates a graph-level
        representation by aggregation type 'avg'.

        The node_feat should have shape :math:`(N\_NODES, F)`.
        Avg_nodes operation will aggregate the node_feat
        according to ver_subgraph_idx.
        The output tensor will have a shape :math:`(N\_GRAPHS, F)`.
        :math:`F` is the dimension of the node feature.

        Args:
            node_feat (Tensor): a tensor represents the node feature,
                with shape :math:`(N\_NODES, F)`.
                :math:`F` is the dimension of the node feature.

        Returns:
            Tensor, a tensor with shape :math:`(N\_GRAPHS, F)`. :math:`F` is the dimension of the node feature.

        Raises:
            TypeError: If `node_feat` is not a Tensor which is the type of operation.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gl import BatchedGraph, BatchedGraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> node_feat = ms.Tensor([
            ...     # graph 1:
            ...     [1, 2, 3, 4],
            ...     [2, 4, 1, 3],
            ...     [1, 3, 2, 4],
            ...     # graph 2:
            ...     [9, 7, 5, 8],
            ...     [8, 7, 6, 5],
            ...     [8, 6, 4, 6],
            ...     [1, 2, 1, 1],
            ... ], ms.float32)
            ...
            >>> n_nodes = 7
            >>> n_edges = 8
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
            >>> ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
            >>> edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
            >>> graph_mask = ms.Tensor([1, 1], ms.int32)
            >>> batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges,
            ...                                         ver_subgraph_idx, edge_subgraph_idx, graph_mask)
            ...
            >>> class TestAvgNodes(GNNCell):
            ...     def construct(self, x, bg: BatchedGraph):
            ...         return bg.avg_nodes(x)
            ...
            >>> ret = TestAvgNodes()(node_feat, *batched_graph_field.get_batched_graph()).asnumpy().tolist()
            >>> print(ret)
                [[1.3333333730697632, 3.0, 2.0, 3.6666667461395264], [6.5, 5.5, 4.0, 5.0]]
        """

    def avg_edges(self, edge_feat):
        r"""
        Aggregating edge features and generates a graph-level
        representation by aggregation type 'avg'.

        The edge_feat should have shape :math:`(N\_EDGES, F)`.
        Avg_edges operation will aggregate the edge_feat
        according to edge_subgraph_idx.
        The output tensor will have a shape :math:`(N\_GRAPHS, F)`.
        :math:`F` is the dimension of the edge feature.

        Args:
            edge_feat (Tensor): a tensor represents the edge feature,
                with shape :math:`(N\_EDGES, F)`. :math:`F` is the dimension of the edge feature.

        Returns:
            Tensor, a tensor with shape :math:`(N\_GRAPHS, F)`,
            :math:`F` is the dimension of the edge feature.

        Raises:
            TypeError: If `edge_feat` is not a Tensor which is the type of operation 'shape'.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gl import BatchedGraph, BatchedGraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> edge_feat = ms.Tensor([
            ...     # graph 1:
            ...     [1, 2, 3, 4],
            ...     [2, 4, 1, 3],
            ...     [1, 3, 2, 4],
            ...     # graph 2:
            ...     [9, 7, 5, 8],
            ...     [8, 7, 6, 5],
            ...     [8, 6, 4, 6],
            ...     [1, 2, 1, 1],
            ...     [3, 2, 3, 3],
            ... ], ms.float32)
            ...
            >>> n_nodes = 7
            >>> n_edges = 8
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
            >>> ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
            >>> edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
            >>> graph_mask = ms.Tensor([1, 1], ms.int32)
            >>> batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges,
            ...                                         ver_subgraph_idx, edge_subgraph_idx, graph_mask)
            ...
            >>> class TestAvgEdges(GNNCell):
            ...     def construct(self, x, bg: BatchedGraph):
            ...         return bg.avg_edges(x)
            ...
            >>> ret = TestAvgEdges()(edge_feat, *batched_graph_field.get_batched_graph()).asnumpy().tolist()
            >>> print(ret)
                [[1.3333333730697632, 3.0, 2.0, 3.6666667461395264],
                 [5.800000190734863, 4.800000190734863, 3.799999952316284, 4.599999904632568]]
        """

    def softmax_nodes(self, node_feat):
        r"""
        Perform graph-wise softmax on the node features.

        For each node :math:`v\in\mathcal{V}` and its feature :math:`x_v`,
        calculate its normalized feature as follows:

        .. math::
            z_v = \frac{\exp(x_v)}{\sum_{u\in\mathcal{V}}\exp(x_u)}

        Each subgraph computes softmax independently.
        The result tensor has the same shape as the original node feature.

        Args:
            node_feat (Tensor): a tensor represent the node feature,
                with shape :math:`(N\_NODES, F)`, :math:`F` is the feature size.

        Returns:
            Tensor, a tensor with shape :math:`(N\_NODES, F)`, :math:`F` is the feature size.

        Raises:
            TypeError: If `node_feat` is not a Tensor.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> import numpy as np
            >>> from mindspore_gl import BatchedGraph, BatchedGraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> node_feat = ms.Tensor([
            ...     # graph 1:
            ...     [1, 2, 3, 4],
            ...     [2, 4, 1, 3],
            ...     [1, 3, 2, 4],
            ...     # graph 2:
            ...     [9, 7, 5, 8],
            ...     [8, 7, 6, 5],
            ...     [8, 6, 4, 6],
            ...     [1, 2, 1, 1],
            ... ], ms.float32)
            ...
            >>> n_nodes = 7
            >>> n_edges = 8
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
            >>> ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
            >>> edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
            >>> graph_mask = ms.Tensor([1, 1], ms.int32)
            >>> batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges,
            ...                                         ver_subgraph_idx, edge_subgraph_idx, graph_mask)
            ...
            >>> class TestSoftmaxNodes(GNNCell):
            ...     def construct(self, x, bg: BatchedGraph):
            ...         return bg.softmax_nodes(x)
            ...
            >>> ret = TestSoftmaxNodes()(node_feat, *batched_graph_field.get_batched_graph()).asnumpy()
            >>> print(np.array2string(ret, formatter={'float_kind':'{0:.5f}'.format}))
                [[0.21194, 0.09003, 0.66524, 0.42232],
                 [0.57612, 0.66524, 0.09003, 0.15536],
                 [0.21194, 0.24473, 0.24473, 0.42232],
                 [0.57601, 0.42112, 0.24364, 0.84315],
                 [0.21190, 0.42112, 0.66227, 0.04198],
                 [0.21190, 0.15492, 0.08963, 0.11411],
                 [0.00019, 0.00284, 0.00446, 0.00077]]
        """

    def softmax_edges(self, edge_feat):
        r"""
        Perform graph-wise softmax on the edge features.

        For each edge :math:`v\in\mathcal{V}` and its feature :math:`x_v`,
        calculate its normalized feature as follows:

        .. math::
            z_v = \frac{\exp(x_v)}{\sum_{u\in\mathcal{V}}\exp(x_u)}

        Each subgraph computes softmax independently.
        The result tensor has the same shape as the original edge feature.

        Args:
            edge_feat (Tensor): a tensor represent the edge feature,
                with shape :math:`(N\_EDGES, F)`, :math:`F` is the feature size.

        Returns:
            Tensor, a tensor with shape :math:`(N\_EDGES, F)`, :math:`F` is the feature size.

        Raises:
            TypeError: If `edge_feat` is not a Tensor.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> import numpy as np
            >>> from mindspore_gl import BatchedGraph, BatchedGraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> edge_feat = ms.Tensor([
            ...     # graph 1:
            ...     [1, 2, 3, 4],
            ...     [2, 4, 1, 3],
            ...     [1, 3, 2, 4],
            ...     # graph 2:
            ...     [9, 7, 5, 8],
            ...     [8, 7, 6, 5],
            ...     [8, 6, 4, 6],
            ...     [1, 2, 1, 1],
            ...     [3, 2, 3, 3],
            ... ], ms.float32)
            ...
            >>> n_nodes = 7
            >>> n_edges = 8
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
            >>> ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
            >>> edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
            >>> graph_mask = ms.Tensor([1, 1], ms.int32)
            >>> batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges,
            ...                                         ver_subgraph_idx, edge_subgraph_idx, graph_mask)
            ...
            >>> class TestSoftmaxEdges(GNNCell):
            ...     def construct(self, x, bg: BatchedGraph):
            ...         return bg.softmax_edges(x)
            ...
            >>> ret = TestSoftmaxEdges()(edge_feat, *batched_graph_field.get_batched_graph()).asnumpy()
            >>> print(np.array2string(ret, formatter={'float_kind':'{0:.5f}'.format}))
                [[0.21194, 0.09003, 0.66524, 0.42232],
                 [0.57612, 0.66524, 0.09003, 0.15536],
                 [0.21194, 0.24473, 0.24473, 0.42232],
                 [0.57518, 0.41993, 0.23586, 0.83838],
                 [0.21160, 0.41993, 0.64113, 0.04174],
                 [0.21160, 0.15448, 0.08677, 0.11346],
                 [0.00019, 0.00283, 0.00432, 0.00076],
                 [0.00143, 0.00283, 0.03192, 0.00565]]
        """

    def broadcast_nodes(self, graph_feat):
        r"""
        Broadcast graph-level features to node-level representation.

        Args:
            graph_feat (Tensor): a tensor represent the graph feature,
                with shape :math:`(N\_GRAPHS, F)`, :math:`F` is the feature size.

        Returns:
            Tensor, a tensor with shape :math:`(N\_NODES, F)`, :math:`F` is the feature size.

        Raises:
            TypeError: If `graph_feat` is not a Tensor.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gl import BatchedGraph, BatchedGraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> node_feat = ms.Tensor([
            ...     # graph 1:
            ...     [1, 2, 3, 4],
            ...     [2, 4, 1, 3],
            ...     [1, 3, 2, 4],
            ...     # graph 2:
            ...     [9, 7, 5, 8],
            ...     [8, 7, 6, 5],
            ...     [8, 6, 4, 6],
            ...     [1, 2, 1, 1],
            ... ], ms.float32)
            ...
            >>> n_nodes = 7
            >>> n_edges = 8
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
            >>> ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
            >>> edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
            >>> graph_mask = ms.Tensor([1, 1], ms.int32)
            >>> batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges,
            ...                                         ver_subgraph_idx, edge_subgraph_idx, graph_mask)
            ...
            >>> class TestBroadCastNodes(GNNCell):
            ...     def construct(self, x, bg: BatchedGraph):
            ...         ret = bg.max_nodes(x)
            ...         return bg.broadcast_nodes(ret)
            ...
            >>> ret = TestBroadCastNodes()(node_feat, *batched_graph_field.get_batched_graph()).asnumpy().tolist()
            >>> print(ret)
                [[2.0, 4.0, 3.0, 4.0], [2.0, 4.0, 3.0, 4.0], [2.0, 4.0, 3.0, 4.0],
                 [9.0, 7.0, 6.0, 8.0], [9.0, 7.0, 6.0, 8.0], [9.0, 7.0, 6.0, 8.0], [9.0, 7.0, 6.0, 8.0]]
        """

    def broadcast_edges(self, graph_feat):
        r"""
        Broadcast graph-level features to edge-level representation.

        Args:
            graph_feat (Tensor): a tensor represent the graph feature,
                with shape :math:`(N\_GRAPHS, F)`, :math:`F` is the feature size.

        Returns:
            Tensor, a tensor with shape :math:`(N\_EDGES, F)`, :math:`F` is the feature size.

        Raises:
            TypeError: If `graph_feat` is not a Tensor.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gl import BatchedGraph, BatchedGraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> edge_feat = ms.Tensor([
            ...     # graph 1:
            ...     [1, 2, 3, 4],
            ...     [2, 4, 1, 3],
            ...     [1, 3, 2, 4],
            ...     # graph 2:
            ...     [9, 7, 5, 8],
            ...     [8, 7, 6, 5],
            ...     [8, 6, 4, 6],
            ...     [1, 2, 1, 1],
            ...     [3, 2, 3, 3],
            ... ], ms.float32)
            ...
            >>> n_nodes = 7
            >>> n_edges = 8
            >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
            >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
            >>> ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
            >>> edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
            >>> graph_mask = ms.Tensor([1, 1], ms.int32)
            >>> batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges,
            ...                                         ver_subgraph_idx, edge_subgraph_idx, graph_mask)
            ...
            >>> class TestBroadCastEdges(GNNCell):
            ...     def construct(self, x, bg: BatchedGraph):
            ...         ret = bg.max_edges(x)
            ...         return bg.broadcast_edges(ret)
            ...
            >>> ret = TestBroadCastEdges()(edge_feat, *batched_graph_field.get_batched_graph()).asnumpy().tolist()
            >>> print(ret)
                [[2.0, 4.0, 3.0, 4.0], [2.0, 4.0, 3.0, 4.0], [2.0, 4.0, 3.0, 4.0],
                 [9.0, 7.0, 6.0, 8.0], [9.0, 7.0, 6.0, 8.0], [9.0, 7.0, 6.0, 8.0],
                 [9.0, 7.0, 6.0, 8.0], [9.0, 7.0, 6.0, 8.0]]
        """


class HeterGraph:
    """
    The heterogeneous Graph.

    This is the class which should be annotated in construct function
    for GNNCell class.
    """

    @property
    def src_idx(self):
        r"""
        A list of tensor with shape :math:`(N\_EDGES)`, represents the source
        node index of COO edge matrix.
        """

    @property
    def dst_idx(self):
        r"""
        A list of tensor with shape :math:`(N\_EDGES)`, represents the
        destination node index of COO edge matrix.
        """

    @property
    def n_nodes(self):
        """
        A list of integer, represent the nodes count of the graph.
        """

    @property
    def n_edges(self):
        """
        A list of integer, represent the edges count of the graph.
        """

    def __init__(self):
        pass

    def get_homo_graph(self, etype):
        """
        Get the specific nodes, edges for etype.

        Args:
            etype (int): The edge type.

        Returns:
            List[Tensor], a homo graph.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gl import Graph, HeterGraph, HeterGraphField
            >>> from mindspore_gl.nn import GNNCell
            >>> n_nodes = [9, 2]
            >>> n_edges = [11, 1]
            >>> src_idx = [ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32), ms.Tensor([0], ms.int32)]
            >>> dst_idx = [ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32), ms.Tensor([1], ms.int32)]
            >>> heter_graph_field = HeterGraphField(src_idx, dst_idx, n_nodes, n_edges)
            >>> node_feat = ms.Tensor([[1], [2], [1], [2], [0], [1], [2], [3], [1]], ms.float32)
            ...
            >>> class TestSum(GNNCell):
            ...     def construct(self, x, g: Graph):
            ...         g.set_vertex_attr({"x": x})
            ...         for v in g.dst_vertex:
            ...             v.h = g.sum([u.x for u in v.innbs])
            ...         return [v.h for v in g.dst_vertex]
            ...
            >>> class TestHeterGraph(GNNCell):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.sum = TestSum()
            ...
            ...     def construct(self, x, hg: HeterGraph):
            ...         return self.sum(x, *hg.get_homo_graph(0))
            ...
            >>> ret = TestHeterGraph()(node_feat, *heter_graph_field.get_heter_graph()).asnumpy().tolist()
            >>> print(ret)
                [[1.0], [2.0], [0.0], [0.0], [3.0], [2.0], [1.0], [0.0], [3.0]]
        """


class GraphField:
    r"""
    The data container for a graph.

    The edge information are stored in COO format.

    Args:
        src_idx (Tensor): A tensor with shape :math:`(N\_EDGES)`, with int dtype,
            represents the source node index of COO edge matrix.
        dst_idx (Tensor): A tensor with shape :math:`(N\_EDGES)`, with int dtype,
            represents the destination node index of COO edge matrix.
        n_nodes (int): An integer, represent the nodes count of the graph.
        n_edges (int): An integer, represent the edges count of the graph.

    Supported Platforms:
            ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore_gl import GraphField
        >>> n_nodes = 9
        >>> n_edges = 11
        >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
        >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
        >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
    """

    def __init__(self, src_idx, dst_idx, n_nodes, n_edges):
        self.src_idx = src_idx
        self.dst_idx = dst_idx
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        if not isinstance(self.src_idx, ms.Tensor):
            raise TypeError(f"src_idx should be a tensor, but got {type(self.src_idx)}.")
        if not isinstance(self.src_idx.dtype, ms.Int):
            raise TypeError(f"src_idx should be an int datatype, but got {self.src_idx.dtype}.")
        if not isinstance(self.dst_idx, ms.Tensor):
            raise TypeError(f"dst_idx should be a tensor, but got {type(self.dst_idx)}.")
        if not isinstance(self.dst_idx.dtype, ms.Int):
            raise TypeError(f"dst_idx should be an int datatype, but got {self.dst_idx.dtype}.")

        if isinstance(self.n_nodes, ms.Tensor):
            if self.n_nodes.dtype == ms.bool_:
                raise TypeError(f"n_nodes should be an integer, but got {self.n_nodes.dtype}.")
            self.n_nodes = int(self.n_nodes.asnumpy())
        if isinstance(self.n_edges, ms.Tensor):
            if self.n_edges.dtype == ms.bool_:
                raise TypeError(f"n_edges should be an integer, but got {self.n_edges.dtype}.")
            self.n_edges = int(self.n_edges.asnumpy())
        if isinstance(self.n_nodes, bool):
            raise TypeError(f"n_nodes should be an integer, but got {type(self.n_nodes)}.")
        if isinstance(self.n_edges, bool):
            raise TypeError(f"n_edges should be an integer, but got {type(self.n_edges)}.")
        if not isinstance(self.n_nodes, int):
            raise TypeError(f"n_nodes should be an integer, but got {type(self.n_nodes)}.")
        if not isinstance(self.n_edges, int):
            raise TypeError(f"n_edges should be an integer, but got {type(self.n_edges)}.")

    def get_graph(self):
        """
        Get the Graph.

        Returns:
            List, A list of tensor, which should be
            used for construct function.
        """
        return [self.src_idx, self.dst_idx, self.n_nodes, self.n_edges]


class BatchedGraphField(GraphField):
    r"""
    The data container for a batched graph.

    The edge information are stored in COO format.

    Args:
        src_idx (Tensor): A tensor with shape :math:`(N\_EDGES)`, with int dtype,
            represents the source node index of COO edge matrix.
        dst_idx (Tensor): A tensor with shape :math:`(N\_EDGES)`, with int dtype,
            represents the destination node index of COO edge matrix.
        n_nodes (int): An integer, represent the nodes count of the graph.
        n_edges (int): An integer, represent the edges count of the graph.
        ver_subgraph_idx (Tensor): A tensor with shape :math:`(N\_NODES)`, with int dtype,
            indicates each node belonging to which subgraph.
        edge_subgraph_idx (Tensor): A tensor with shape :math:`(N\_EDGES,)`, with int dtype,
            indicates each edge belonging to which subgraph.
        graph_mask (Tensor): A tensor with shape :math:`(N\_GRAPHS,)`, with int dtype,
            indicates whether the subgraph is exist.

    Supported Platforms:
            ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore_gl import BatchedGraphField
        >>> n_nodes = 9
        >>> n_edges = 11
        >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
        >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
        >>> ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 2, 2], ms.int32)
        >>> edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2], ms.int32)
        >>> graph_mask = ms.Tensor([1, 1, 0], ms.int32)
        >>> batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges,
        ...                                         ver_subgraph_idx, edge_subgraph_idx, graph_mask)
    """

    def __init__(self, src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx,
                 edge_subgraph_idx, graph_mask):
        super().__init__(src_idx, dst_idx, n_nodes, n_edges)
        self.ver_subgraph_idx = ver_subgraph_idx
        self.edge_subgraph_idx = edge_subgraph_idx
        self.graph_mask = graph_mask
        if not isinstance(self.ver_subgraph_idx, ms.Tensor):
            raise TypeError(f"ver_subgraph_idx should be a tensor, but got {type(self.ver_subgraph_idx)}.")
        if not isinstance(self.ver_subgraph_idx.dtype, ms.Int):
            raise TypeError(f"ver_subgraph_idx should be an int datatype, but got {self.ver_subgraph_idx.dtype}.")
        if not isinstance(self.edge_subgraph_idx, ms.Tensor):
            raise TypeError(f"edge_subgraph_idx should be a tensor, but got {type(self.edge_subgraph_idx)}.")
        if not isinstance(self.edge_subgraph_idx.dtype, ms.Int):
            raise TypeError(f"edge_subgraph_idx should be an int datatype, but got {self.edge_subgraph_idx.dtype}.")
        if not isinstance(self.graph_mask, ms.Tensor):
            raise TypeError(f"graph_mask should be a tensor, but got {type(self.graph_mask)}.")
        if not isinstance(self.graph_mask.dtype, ms.Int):
            raise TypeError(f"graph_mask should be an int datatype, but got {self.graph_mask.dtype}.")

    def get_batched_graph(self):
        """
        Get the batched Graph.

        Returns:
            List, A list of tensor, which should
            be used for construct function.
        """
        batched_graph_field = self.get_graph()
        batched_graph_field.extend(
            [self.ver_subgraph_idx,
             self.edge_subgraph_idx,
             self.graph_mask])
        return batched_graph_field


class HeterGraphField:
    r"""
    The data container for a heterogeneous graph.
    The edge information are stored in COO format.

    Args:
        src_idx (List[Tensor]): A list of tensor with shape :math:`(N\_EDGES)`, with int dtype,
            represents the source node index of COO edge matrix.
        dst_idx (List[Tensor]): A list of tensor with shape :math:`(N\_EDGES)`, with int dtype,
            represents the destination node index of COO edge matrix.
        n_nodes (List[int]): A list of integer, represent the nodes count of the graph.
        n_edges (List[int]): A list of integer, represent the edges count of the graph.

    Supported Platforms:
            ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore_gl import HeterGraphField
        >>> n_nodes = [9, 2]
        >>> n_edges = [11, 1]
        >>> src_idx = [ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32), ms.Tensor([0], ms.int32)]
        >>> dst_idx = [ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32), ms.Tensor([1], ms.int32)]
        >>> heter_graph_field = HeterGraphField(src_idx, dst_idx, n_nodes, n_edges)
    """

    def __init__(self, src_idx, dst_idx, n_nodes, n_edges):
        self.src_idx = src_idx
        self.dst_idx = dst_idx
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        if not isinstance(self.src_idx, list):
            raise TypeError(f"src_idx should be a list, but got {type(self.src_idx)}.")
        for per_src_idx in self.src_idx:
            if not isinstance(per_src_idx, ms.Tensor):
                raise TypeError(f"element of src_idx should be a tensor, but got {type(per_src_idx)}.")
            if not isinstance(per_src_idx.dtype, ms.Int):
                raise TypeError(f"element of src_idx should be an int datatype, but got {per_src_idx.dtype}.")
        if not isinstance(self.dst_idx, list):
            raise TypeError(f"dst_idx should be a list, but got {type(self.dst_idx)}.")
        for per_dst_idx in self.dst_idx:
            if not isinstance(per_dst_idx, ms.Tensor):
                raise TypeError(f"element of dst_idx should be a tensor, but got {type(per_dst_idx)}.")
            if not isinstance(per_dst_idx.dtype, ms.Int):
                raise TypeError(f"element of dst_idx should be an int datatype, but got {per_dst_idx.dtype}.")
        if not isinstance(self.n_nodes, list):
            raise TypeError(f"n_nodes should be a list, but got {type(self.n_nodes)}.")
        if not isinstance(self.n_edges, list):
            raise TypeError(f"n_edges should be a list, but got {type(self.n_edges)}.")

    def get_heter_graph(self):
        """
        Get the hetergenous Graph.

        Returns:
            List, A list of tensor list, which should be used for construct
            function.
        """
        return [self.src_idx, self.dst_idx, self.n_nodes, self.n_edges]
