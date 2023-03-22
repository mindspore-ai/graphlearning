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
""" test api """
import math
import numpy as np
import mindspore as ms
import mindspore.context as context
import pytest
from mindspore_gl import Graph, BatchedGraph, HeterGraph, GraphField, BatchedGraphField, HeterGraphField
from mindspore_gl.nn import GNNCell


@pytest.fixture(name="setup", autouse=True, scope='module')
def fixture_setup():
    context.set_context(device_target="GPU", mode=context.GRAPH_MODE)
    yield


@pytest.fixture(name="node_feat", autouse=True, scope='module')
def fixture_node_feat():
    yield ms.Tensor([[1], [2], [1], [2], [0], [1], [2], [3], [1]], ms.float32)


@pytest.fixture(name="edge_feat", autouse=True, scope='module')
def fixture_edge_feat():
    yield ms.Tensor([[1], [2], [1], [3], [1], [4], [1], [5], [1], [1], [1]], ms.float32)


@pytest.fixture(name="graph_field", autouse=True, scope='module')
def fixture_graph_field():
    n_nodes = 9
    n_edges = 11
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
    yield GraphField(src_idx, dst_idx, n_nodes, n_edges)


@pytest.fixture(name="heter_graph_field", autouse=True, scope='module')
def fixture_heter_graph_field():
    n_nodes = [9, 2]
    n_edges = [11, 1]
    src_idx = [ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32), ms.Tensor([0], ms.int32)]
    dst_idx = [ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32), ms.Tensor([1], ms.int32)]
    yield HeterGraphField(src_idx, dst_idx, n_nodes, n_edges)


@pytest.fixture(name="batched_graph_field", autouse=True, scope='module')
def fixture_batched_graph_field():
    n_nodes = 9
    n_edges = 11
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 2, 2], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2], ms.int32)
    graph_mask = ms.Tensor([1, 1, 0], ms.int32)
    yield BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx, edge_subgraph_idx, graph_mask)


def assert_list_equal(first: list, second: list):
    """Helper function"""
    first, second = np.array(first), np.array(second)
    assert np.sum(np.abs((first - second))) == 0


def assert_list_almost_equal(first: list, second: list, delta: float = 1e-5):
    """Helper function"""
    first, second = np.array(first), np.array(second)
    assert np.max(np.abs((first - second))) < delta


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_graph_property(graph_field):
    """
    Feature:test graph property
    Description:Init a Graph
    n_nodes = 9
    n_edges = 11
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
    Expectation:results == [src_idx, dst_idx, n_nodes, n_edges]
    """

    class TestProperty(GNNCell):
        def construct(self, g: Graph):
            return [g.src_idx, g.dst_idx, g.n_nodes, g.n_edges]

    ret = TestProperty()(*graph_field.get_graph())
    expected = graph_field.get_graph()
    assert_list_equal(ret[0].asnumpy().tolist(), expected[0].asnumpy().tolist())
    assert_list_equal(ret[1].asnumpy().tolist(), expected[1].asnumpy().tolist())
    assert ret[2] == expected[2]
    assert ret[3] == expected[3]


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batched_graph_property(batched_graph_field):
    """
    Feature: test batched graph property
    Description:Init batched Graph
    n_nodes = 9
    n_edges = 11
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 2, 2], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2], ms.int32)
    graph_mask = ms.Tensor([1, 1, 0], ms.int32)
    Expectation:results == [src_idx, dst_idx, n_nodes, n_edges,
                    ver_subgraph_idx, edge_subgraph_idx, graph_mask, n_graphs]
    """

    class TestProperty(GNNCell):
        def construct(self, g: BatchedGraph):
            return [g.src_idx, g.dst_idx, g.n_nodes, g.n_edges,
                    g.ver_subgraph_idx, g.edge_subgraph_idx, g.graph_mask, g.n_graphs]

    ret = TestProperty()(*batched_graph_field.get_batched_graph())
    expected = batched_graph_field.get_batched_graph()
    assert_list_equal(ret[0].asnumpy().tolist(), expected[0].asnumpy().tolist())
    assert_list_equal(ret[1].asnumpy().tolist(), expected[1].asnumpy().tolist())
    assert ret[2] == expected[2]
    assert ret[3] == expected[3]
    assert_list_equal(ret[4].asnumpy().tolist(), expected[4].asnumpy().tolist())
    assert_list_equal(ret[5].asnumpy().tolist(), expected[5].asnumpy().tolist())
    assert_list_equal(ret[6].asnumpy().tolist(), expected[6].asnumpy().tolist())
    assert ret[7] == 3


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_adj_to_dense(graph_field):
    """
    Feature: test adj_to_dense
    Description:init Coo format graph
    n_nodes = 9
    n_edges = 11
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
    Expectation: dense matrix graph
    [
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 3],
    ]
    """

    class TestAdjToDense(GNNCell):
        def construct(self, g: Graph):
            return g.adj_to_dense()

    ret = TestAdjToDense()(*graph_field.get_graph()).asnumpy().tolist()
    expected = [
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 3],
    ]
    assert_list_equal(ret, expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_set_vertex_attr(node_feat, graph_field):
    """
     Feature: test set_vertex_attr
     Description: init a graph and input vertex feat
     graph:
     n_nodes = 9
     n_edges = 11
     src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
     dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
     node_feat:ms.Tensor([[1], [2], [1], [2], [0], [1], [2], [3], [1]], ms.float32)
     Expectation: v.h*u.h
      [[1], [4], [1], [4], [0], [1], [4], [9], [1]]
     """

    class TestSetVertexAttr(GNNCell):
        def construct(self, x, g: Graph):
            g.set_vertex_attr({"h": x})
            return [v.h for v in g.dst_vertex] * [u.h for u in g.src_vertex]

    ret = TestSetVertexAttr()(node_feat, *graph_field.get_graph()).asnumpy().tolist()
    expected = [[1], [4], [1], [4], [0], [1], [4], [9], [1]]
    assert_list_equal(ret, expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_set_src_attr(node_feat, graph_field):
    """
    Feature: test set_src_attr
    Description: init a graph and input src vertex feat
    graph:
    n_nodes = 9
    n_edges = 11
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
    node_feat:ms.Tensor([[1], [2], [1], [2], [0], [1], [2], [3], [1]], ms.float32)
    Expectation:u.h
    [[1], [2], [1], [2], [0], [1], [2], [3], [1]]
    """

    class TestSetSrcAttr(GNNCell):
        def construct(self, x, g: Graph):
            g.set_src_attr({"h": x})
            return [u.h for u in g.src_vertex]

    ret = TestSetSrcAttr()(node_feat, *graph_field.get_graph()).asnumpy().tolist()
    expected = [[1], [2], [1], [2], [0], [1], [2], [3], [1]]
    assert_list_equal(ret, expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_set_dst_attr(node_feat, graph_field):
    """
    Feature: test set_dst_attr
    Description: init a graph and input dst vertex feat
    graph:
    n_nodes = 9
    n_edges = 11
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
    node_feat:ms.Tensor([[1], [2], [1], [2], [0], [1], [2], [3], [1]], ms.float32)
    Expectation:v.h
    [[1], [2], [1], [2], [0], [1], [2], [3], [1]]
    """

    class TestSetDstAttr(GNNCell):
        def construct(self, x, g: Graph):
            g.set_dst_attr({"h": x})
            return [v.h for v in g.dst_vertex]

    ret = TestSetDstAttr()(node_feat, *graph_field.get_graph()).asnumpy().tolist()
    expected = [[1], [2], [1], [2], [0], [1], [2], [3], [1]]
    assert_list_equal(ret, expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_set_edge_attr(node_feat, edge_feat, graph_field):
    """
    Feature: test set_edge_attr
    Description: init a graph and input vertex feat and edge feat
    graph:
    n_nodes = 9
    n_edges = 11
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
    node_feat:ms.Tensor([[1], [2], [1], [2], [0], [1], [2], [3], [1]], ms.float32)
    edge_feat:ms.Tensor([[1], [2], [1], [3], [1], [4], [1], [5], [1], [1], [1]], ms.float32)
    Expectation:v.h = sum(nh*eh)
    [[2], [2], [0], [0], [14], [6], [1], [0], [3]]
    """

    class TestSetEdgeAttr(GNNCell):
        def construct(self, nh, eh, g: Graph):
            g.set_vertex_attr({"nh": nh})
            g.set_edge_attr({"eh": eh})
            for v in g.dst_vertex:
                v.h = g.sum([u.nh * e.eh for u, e in v.inedges])
            return [v.h for v in g.dst_vertex]

    ret = TestSetEdgeAttr()(node_feat, edge_feat, *graph_field.get_graph()).asnumpy().tolist()
    expected = [[2], [2], [0], [0], [14], [6], [1], [0], [3]]
    assert_list_equal(ret, expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dot(node_feat, graph_field):
    """
    Feature: test dot
    Description: init a graph and src feat and dst feat
    graph:
    n_nodes = 9
    n_edges = 11
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
    node_feat: ms.Tensor([[1], [2], [1], [2], [0], [1], [2], [3], [1]], ms.float32)
    Expectation:dot(v, v.innbs)
    [[2], [1], [2], [2], [0], [0], [2], [0], [1], [1], [1]]
    """

    class TestDot(GNNCell):
        def construct(self, x, g: Graph):
            g.set_vertex_attr({"src": x, "dst": x})
            for v in g.dst_vertex:
                v.h = [g.dot(v.src, u.dst) for u in v.innbs]
            return [v.h for v in g.dst_vertex]

    ret = TestDot()(node_feat, *graph_field.get_graph()).asnumpy().tolist()
    expected = [[2], [1], [2], [2], [0], [0], [2], [0], [1], [1], [1]]
    assert_list_equal(ret, expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sum(node_feat, graph_field):
    """
    Feature: test sum
    Description: init a graph and src feat
    graph:
    n_nodes = 9
    n_edges = 11
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
    node_feat: ms.Tensor([[1], [2], [1], [2], [0], [1], [2], [3], [1]], ms.float32)
    Expectation:sum([v.innbs])
    [[1], [2], [0], [0], [3], [2], [1], [0], [3]]
    """

    class TestSum(GNNCell):
        def construct(self, x, g: Graph):
            g.set_vertex_attr({"x": x})
            for v in g.dst_vertex:
                v.h = g.sum([u.x for u in v.innbs])
            return [v.h for v in g.dst_vertex]

    ret = TestSum()(node_feat, *graph_field.get_graph()).asnumpy().tolist()
    expected = [[1], [2], [0], [0], [3], [2], [1], [0], [3]]
    assert_list_equal(ret, expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_max(node_feat, graph_field):
    """
    Feature: test max
    Description: init a graph and src feat
    graph:
    n_nodes = 9
    n_edges = 11
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
    node_feat: ms.Tensor([[1], [2], [1], [2], [0], [1], [2], [3], [1]], ms.float32)
    Expectation:max([v.innbs])
    [[1], [1], [0], [0], [2], [2], [1], [0], [1]]
    """

    class TestMax(GNNCell):
        def construct(self, x, g: Graph):
            g.set_vertex_attr({"x": x})
            for v in g.dst_vertex:
                v.h = g.max([u.x for u in v.innbs])
            return [v.h for v in g.dst_vertex]

    ret = TestMax()(node_feat, *graph_field.get_graph()).asnumpy().tolist()
    expected = [[1], [1], [0], [0], [2], [2], [1], [0], [1]]
    assert_list_equal(ret, expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_min(node_feat, graph_field):
    """
    Feature: test min
    Description: init a graph and src feat
    graph:
    n_nodes = 9
    n_edges = 11
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
    node_feat: ms.Tensor([[1], [2], [1], [2], [0], [1], [2], [3], [1]], ms.float32)
    Expectation:min([v.innbs])
    [[1], [1], [0], [0], [1], [2], [1], [0], [1]]
    """

    class TestMin(GNNCell):
        def construct(self, x, g: Graph):
            g.set_vertex_attr({"x": x})
            for v in g.dst_vertex:
                v.h = g.min([u.x for u in v.innbs])
            return [v.h for v in g.dst_vertex]

    ret = TestMin()(node_feat, *graph_field.get_graph()).asnumpy().tolist()
    expected = [[1], [1], [0], [0], [1], [2], [1], [0], [1]]
    assert_list_equal(ret, expected)


NAN = 1e9


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_avg(node_feat, graph_field):
    """
    Feature: test avg
    Description: init a graph and src feat
    graph:
    n_nodes = 9
    n_edges = 11
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
    node_feat: ms.Tensor([[1], [2], [1], [2], [0], [1], [2], [3], [1]], ms.float32)
    Expectation:min([v.innbs])
    [[1], [1], [0], [0], [1.5], [2], [1], [0], [1]]
    """

    class TestAvg(GNNCell):
        def construct(self, x, g: Graph):
            g.set_vertex_attr({"x": x})
            for v in g.dst_vertex:
                v.h = g.avg([u.x for u in v.innbs])
            return [v.h for v in g.dst_vertex]

    ret = TestAvg()(node_feat, *graph_field.get_graph()).asnumpy().tolist()

    for row in ret:
        if math.isnan(row[0]):
            row[0] = NAN
    expected = [[1], [1], [0], [0], [1.5], [2], [1], [0], [1]]
    assert_list_equal(ret, expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_node_mask(batched_graph_field):
    """
    Feature: test node_mask
    Description: init a batched graph
    batched graph:
    n_nodes = 9
    n_edges = 11
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 2, 2], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2], ms.int32)
    graph_mask = ms.Tensor([1, 1, 0], ms.int32)
    Expectation:node_mask
    [1, 1, 1, 1, 1, 1, 1, 0, 0]
    """

    class TestNodeMask(GNNCell):
        def construct(self, bg: BatchedGraph):
            return bg.node_mask()

    ret = TestNodeMask()(*batched_graph_field.get_batched_graph()).asnumpy().tolist()
    expected = [1, 1, 1, 1, 1, 1, 1, 0, 0]
    assert_list_equal(ret, expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_edge_mask(batched_graph_field):
    """
    Feature: test edge_mask
    Description: init a batched graph
    batched graph:
    n_nodes = 9
    n_edges = 11
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 2, 2], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2], ms.int32)
    graph_mask = ms.Tensor([1, 1, 0], ms.int32)
    Expectation:edge_mask
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    """

    class TestEdgeMask(GNNCell):
        def construct(self, bg: BatchedGraph):
            return bg.edge_mask()

    ret = TestEdgeMask()(*batched_graph_field.get_batched_graph()).asnumpy().tolist()
    expected = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    assert_list_equal(ret, expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_in_degree(graph_field):
    """
    Feature: test in_degree
    Description: init a graph
    graph:
    n_nodes = 9
    n_edges = 11
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
    Expectation:in_degree of dst vertex
    [[1], [2], [0], [1], [2], [1], [1], [0], [3]]
    """

    class TestInDegree(GNNCell):
        def construct(self, g: Graph):
            return g.in_degree()

    ret = TestInDegree()(*graph_field.get_graph()).asnumpy().tolist()
    expected = [[1], [2], [0], [1], [2], [1], [1], [0], [3]]
    assert_list_equal(ret, expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_out_degree(graph_field):
    """
    Feature: test out_degree
    Description: init a graph
    graph:
    n_nodes = 9
    n_edges = 11
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
    Expectation:out_degree of src vertex
    [[1], [0], [2], [1], [1], [2], [1], [0], [3]]
    """

    class TestOutDegree(GNNCell):
        def construct(self, g: Graph):
            return g.out_degree()

    ret = TestOutDegree()(*graph_field.get_graph()).asnumpy().tolist()
    expected = [[1], [0], [2], [1], [1], [2], [1], [0], [3]]
    assert_list_equal(ret, expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_num_of_nodes(batched_graph_field):
    """
    Feature: test num_of_nodes of subgraph
    Description: init a batched graph
    batched graph:
    n_nodes = 9
    n_edges = 11
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 2, 2], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2], ms.int32)
    graph_mask = ms.Tensor([1, 1, 0], ms.int32)
    Expectation:num_of_nodes of subgraphs
    [[3], [4], [2]]
    """

    class TestNumOfNodes(GNNCell):
        def construct(self, bg: BatchedGraph):
            return bg.num_of_nodes()

    ret = TestNumOfNodes()(*batched_graph_field.get_batched_graph()).asnumpy().tolist()
    expected = [[3], [4], [2]]
    assert_list_equal(ret, expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_num_of_edges(batched_graph_field):
    """
    Feature: test num_of_edges of subgraph
    Description: init a batched graph
    batched graph:
    n_nodes = 9
    n_edges = 11
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 2, 2], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2], ms.int32)
    graph_mask = ms.Tensor([1, 1, 0], ms.int32)
    Expectation:num_of_edges of subgraphs
    [[3], [5], [3]]
    """

    class TestNumOfEdges(GNNCell):
        def construct(self, bg: BatchedGraph):
            return bg.num_of_edges()

    ret = TestNumOfEdges()(*batched_graph_field.get_batched_graph()).asnumpy().tolist()
    expected = [[3], [5], [3]]
    assert_list_equal(ret, expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_get_heter_graph(heter_graph_field):
    """
    Feature: test get_heter_graph
    Description: init a hetero graph
    hetero graph:
    n_nodes = [9, 2]
    n_edges = [11, 1]
    src_idx = [ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32), ms.Tensor([0], ms.int32)]
    dst_idx = [ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32), ms.Tensor([1], ms.int32)]
    Expectation: heterograph
    [src_idx, dst_idx, n_nodes, n_edges]
    """

    class TestHeterGraph(GNNCell):
        def construct(self, hg: HeterGraph):
            return [hg.src_idx[1], hg.dst_idx[1], hg.n_nodes[1], hg.n_edges[1]]

    ret = TestHeterGraph()(*heter_graph_field.get_heter_graph())
    assert ret[2] == 2
    assert ret[3] == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_get_homo_graph(node_feat, heter_graph_field):
    """
    Feature: test get_homo_graph
    Description: init a hetero graph
    hetero graph:
    n_nodes = [9, 2]
    n_edges = [11, 1]
    src_idx = [ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32), ms.Tensor([0], ms.int32)]
    dst_idx = [ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32), ms.Tensor([1], ms.int32)]
    Expectation: sum(v.innbs) of homograph
    [[1], [2], [0], [0], [3], [2], [1], [0], [3]]
    """

    class TestSum(GNNCell):
        def construct(self, x, g: Graph):
            g.set_vertex_attr({"x": x})
            for v in g.dst_vertex:
                v.h = g.sum([u.x for u in v.innbs])
            return [v.h for v in g.dst_vertex]

    class TestHeterGraph(GNNCell):
        def __init__(self):
            super().__init__()
            self.sum = TestSum()

        def construct(self, x, hg: HeterGraph):
            return self.sum(x, *hg.get_homo_graph(0))

    ret = TestHeterGraph()(node_feat, *heter_graph_field.get_heter_graph()).asnumpy().tolist()
    expected = [[1], [2], [0], [0], [3], [2], [1], [0], [3]]
    assert_list_equal(ret, expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_topk_nodes_without_softby():
    """
    Feature:test topk nodes without softby
    Description:test topk nodes without softby
    Expectation:topk_nodes_without_softby of node_feat based on batched_graph_field
    """

    node_feat = ms.Tensor([
        [1, 2, 3, 4],
        [2, 4, 1, 3],
        [1, 3, 2, 4],
        [9, 7, 5, 8],
        [8, 7, 6, 5],
        [8, 6, 4, 6],
        [1, 2, 1, 1],
    ], ms.float32)

    n_nodes = 7
    n_edges = 8
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
    graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)

    class TestTopkNodes(GNNCell):
        def construct(self, x, g: Graph):
            return g.topk_nodes(x, 2)

    output, indices = TestTopkNodes()(node_feat, *graph_field.get_graph())
    output = output.asnumpy().tolist()
    indices = indices.asnumpy().tolist()
    output_expected = [[9, 7, 6, 8], [8, 7, 5, 6]]
    indices_expected = [[3, 3, 4, 3], [4, 4, 3, 5]]
    first, second = np.array(output), np.array(output_expected)
    assert np.sum(np.abs((first - second))) == 0
    first, second = np.array(indices), np.array(indices_expected)
    assert np.sum(np.abs((first - second))) == 0


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_topk_edges_without_softby():
    """
    Feature:test topk edges without softby
    Description:test topk edges without softby
    Expectation:topk_edges_without_softby of edge_feat based on batched_graph_field
    """

    edge_feat = ms.Tensor([
        [1, 2, 3, 4],
        [2, 4, 1, 3],
        [1, 3, 2, 4],
        [9, 7, 5, 8],
        [8, 7, 6, 5],
        [8, 6, 4, 6],
        [1, 2, 1, 1],
        [3, 2, 3, 3],
    ], ms.float32)

    n_nodes = 7
    n_edges = 8
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
    graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)

    class TestTopkEdges(GNNCell):
        def construct(self, x, g: Graph):
            return g.topk_edges(x, 2)

    output, indices = TestTopkEdges()(edge_feat, *graph_field.get_graph())
    output = output.asnumpy().tolist()
    indices = indices.asnumpy().tolist()
    output_expected = [[9, 7, 6, 8], [8, 7, 5, 6]]
    indices_expected = [[3, 3, 4, 3], [4, 4, 3, 5]]
    first, second = np.array(output), np.array(output_expected)
    assert np.sum(np.abs((first - second))) == 0
    first, second = np.array(indices), np.array(indices_expected)
    assert np.sum(np.abs((first - second))) == 0


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_topk_nodes_with_softby():
    """
    Feature:test topk nodes with softby
    Description:test topk nodes with softby
    Expectation:topk_nodes_with_softby of node_feat based on batched_graph_field
    """

    node_feat = ms.Tensor([
        [1, 2, 3, 4],
        [2, 4, 1, 3],
        [1, 3, 2, 4],
        [9, 7, 5, 8],
        [8, 7, 6, 5],
        [8, 6, 4, 6],
        [1, 2, 1, 1],
    ], ms.float32)

    n_nodes = 7
    n_edges = 8
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
    graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)

    class TestTopkNodes(GNNCell):
        def construct(self, x, g: Graph):
            return g.topk_nodes(x, 2, 1)

    output, indices = TestTopkNodes()(node_feat, *graph_field.get_graph())
    output = output.asnumpy().tolist()
    indices = indices.asnumpy().tolist()
    output_expected = [[9, 7, 5, 8], [8, 7, 6, 5]]
    indices_expected = [3, 4]
    first, second = np.array(output), np.array(output_expected)
    assert np.sum(np.abs((first - second))) == 0
    first, second = np.array(indices), np.array(indices_expected)
    assert np.sum(np.abs((first - second))) == 0


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_topk_edges_with_softby():
    """
    Feature:test topk edges with softby
    Description:test topk edges with softby
    Expectation:topk_edges_with_softby of edge_feat based on batched_graph_field
    """

    edge_feat = ms.Tensor([
        [1, 2, 3, 4],
        [2, 4, 1, 3],
        [1, 3, 2, 4],
        [9, 7, 5, 8],
        [8, 7, 6, 5],
        [8, 6, 4, 6],
        [1, 2, 1, 1],
        [3, 2, 3, 3],
    ], ms.float32)

    n_nodes = 7
    n_edges = 8
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
    graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)

    class TestTopkEdges(GNNCell):
        def construct(self, x, g: Graph):
            return g.topk_edges(x, 2, 1)

    output, indices = TestTopkEdges()(edge_feat, *graph_field.get_graph())
    output = output.asnumpy().tolist()
    indices = indices.asnumpy().tolist()
    output_expected = [[9, 7, 5, 8], [8, 7, 6, 5]]
    indices_expected = [3, 4]
    first, second = np.array(output), np.array(output_expected)
    assert np.sum(np.abs((first - second))) == 0
    first, second = np.array(indices), np.array(indices_expected)
    assert np.sum(np.abs((first - second))) == 0

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_csr_graph():
    """
    Feature:test topk edges with softby
    Description:test topk edges with softby
    Expectation:topk_edges_with_softby of edge_feat based on batched_graph_field
    """
    n_nodes = 7
    n_edges = 8
    indices = ms.Tensor([2, 3, 5, 6, 3, 4, 0, 6], ms.int32)
    indptr = ms.Tensor([0, 2, 4, 5, 6, 7, 8, 8], ms.int32)
    indices_backward = ms.Tensor([4, 0, 0, 2, 3, 1, 1, 5], ms.int32)
    indptr_backward = ms.Tensor([0, 1, 1, 2, 4, 5, 6, 8], ms.int32)
    graph_field = GraphField(n_nodes=n_nodes, n_edges=n_edges, indices=indices, indptr=indptr,
                             indices_backward=indices_backward, indptr_backward=indptr_backward, csr=True)
    GNNCell.sparse_compute(True, True)
    class TestCsrGraph(GNNCell):
        def construct(self, g: Graph):
            return g.indices, g.indptr, g.indices_backward, g.indices_backward, g.n_nodes, g.n_edges
    res_indices, res_indptr, res_indices_backward, res_indptr_backward, res_n_nodes, res_n_edges =\
        TestCsrGraph()(*graph_field.get_graph())
    GNNCell.sparse_compute(False, False)
    assert np.sum(np.abs((n_nodes - res_n_nodes))) == 0
    assert np.sum(np.abs((n_edges - res_n_edges))) == 0
    assert np.sum(np.abs((res_indices.asnumpy() - indices.asnumpy()))) == 0
    assert np.sum(np.abs((res_indptr.asnumpy() - indptr.asnumpy()))) == 0
    assert np.sum(np.abs((res_indices_backward.asnumpy() - indices_backward.asnumpy()))) == 0
    assert np.sum(np.abs((res_indptr_backward.asnumpy() - indices_backward.asnumpy()))) == 0

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_csr_batchedgraph():
    """
    Feature:test topk edges with softby
    Description:test topk edges with softby
    Expectation:topk_edges_with_softby of edge_feat based on batched_graph_field
    """
    n_nodes = 7
    n_edges = 8
    indices = ms.Tensor([2, 3, 5, 6, 3, 4, 0, 6], ms.int32)
    indptr = ms.Tensor([0, 2, 4, 5, 6, 7, 8, 8], ms.int32)
    indices_backward = ms.Tensor([4, 0, 0, 2, 3, 1, 1, 5], ms.int32)
    indptr_backward = ms.Tensor([0, 1, 1, 2, 4, 5, 6, 8], ms.int32)
    node_map_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
    edge_map_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
    graph_mask = ms.Tensor([1, 0], ms.int32)
    graph_field = BatchedGraphField(indices=indices, indptr=indptr, indices_backward=indices_backward,
                                    indptr_backward=indptr_backward, csr=True, n_nodes=n_nodes,
                                    n_edges=n_edges, ver_subgraph_idx=node_map_idx,
                                    edge_subgraph_idx=edge_map_idx, graph_mask=graph_mask)
    GNNCell.sparse_compute(True, True)
    class TestCsrBatchedGraph(GNNCell):
        def construct(self, g: BatchedGraph):
            return g.indices, g.indptr, g.indices_backward, g.indices_backward, g.n_nodes, g.n_edges,\
                   g.ver_subgraph_idx, g.edge_subgraph_idx, g.graph_mask
    res_indices, res_indptr, res_indices_backward, res_indptr_backward, res_n_nodes, res_n_edges, res_ver_subgraph_idx,\
    res_edge_subgraph_idx, res_graph_mask = TestCsrBatchedGraph()(*graph_field.get_batched_graph())
    GNNCell.sparse_compute(False, False)
    assert np.sum(np.abs((n_nodes - res_n_nodes))) == 0
    assert np.sum(np.abs((n_edges - res_n_edges))) == 0
    assert np.sum(np.abs((res_indices.asnumpy() - indices.asnumpy()))) == 0
    assert np.sum(np.abs((res_indptr.asnumpy() - indptr.asnumpy()))) == 0
    assert np.sum(np.abs((res_indices_backward.asnumpy() - indices_backward.asnumpy()))) == 0
    assert np.sum(np.abs((res_indptr_backward.asnumpy() - indices_backward.asnumpy()))) == 0
    assert np.sum(np.abs((res_indptr.asnumpy() - indptr.asnumpy()))) == 0
    assert np.sum(np.abs((res_indices_backward.asnumpy() - indices_backward.asnumpy()))) == 0
    assert np.sum(np.abs((res_indptr_backward.asnumpy() - indices_backward.asnumpy()))) == 0
    assert np.sum(np.abs((node_map_idx.asnumpy() - res_ver_subgraph_idx.asnumpy()))) == 0
    assert np.sum(np.abs((edge_map_idx.asnumpy() - res_edge_subgraph_idx.asnumpy()))) == 0
    assert np.sum(np.abs((graph_mask.asnumpy() - res_graph_mask.asnumpy()))) == 0
    