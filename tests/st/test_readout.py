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
""" test readout """
import numpy as np
import mindspore as ms
import mindspore.context as context
import pytest
from mindspore_gl.nn import GNNCell

from mindspore_gl import BatchedGraphField, BatchedGraph

context.set_context(device_target="GPU", mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sum_nodes():
    """
    Feature:test sum nodes
    Description:test sum nodes
    Expectation:sum_nodes of node_feat based on batched_graph_field
    """

    node_feat = ms.Tensor([
        # graph 1:
        [1, 2, 3, 4],
        [2, 4, 1, 3],
        [1, 3, 2, 4],
        # graph 2:
        [9, 7, 5, 8],
        [8, 7, 6, 5],
        [8, 6, 4, 6],
        [1, 2, 1, 1],
    ], ms.float32)

    n_nodes = 7
    n_edges = 8
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
    graph_mask = ms.Tensor([1, 1], ms.int32)
    batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx, edge_subgraph_idx,
                                            graph_mask)

    class TestSumNodes(GNNCell):
        def construct(self, x, bg: BatchedGraph):
            return bg.sum_nodes(x)

    ret = TestSumNodes()(node_feat, *batched_graph_field.get_batched_graph()).asnumpy().tolist()
    expected = [
        [4, 9, 6, 11],
        [26, 22, 16, 20]
    ]

    first, second = np.array(ret), np.array(expected)
    assert np.sum(np.abs((first - second))) == 0


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sum_edges():
    """
    Feature:test sum edges
    Description:test sum edges
    Expectation:sum_edges of edge_feat based on batched_graph_field
    """

    edge_feat = ms.Tensor([
        # graph 1:
        [1, 2, 3, 4],
        [2, 4, 1, 3],
        [1, 3, 2, 4],
        # graph 2:
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
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
    graph_mask = ms.Tensor([1, 1], ms.int32)
    batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx, edge_subgraph_idx,
                                            graph_mask)

    class TestSumEdges(GNNCell):
        def construct(self, x, bg: BatchedGraph):
            return bg.sum_edges(x)

    ret = TestSumEdges()(edge_feat, *batched_graph_field.get_batched_graph()).asnumpy().tolist()
    expected = [
        [4, 9, 6, 11],
        [29, 24, 19, 23]
    ]
    first, second = np.array(ret), np.array(expected)
    assert np.sum(np.abs((first - second))) == 0


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_max_nodes():
    """
    Feature:test max nodes
    Description:test max nodes
    Expectation:max_nodes of node_feat based on batched_graph_field
    """

    node_feat = ms.Tensor([
        # graph 1:
        [1, 2, 3, 4],
        [2, 4, 1, 3],
        [1, 3, 2, 4],
        # graph 2:
        [9, 7, 5, 8],
        [8, 7, 6, 5],
        [8, 6, 4, 6],
        [1, 2, 1, 1],
    ], ms.float32)

    n_nodes = 7
    n_edges = 8
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
    graph_mask = ms.Tensor([1, 1], ms.int32)
    batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx, edge_subgraph_idx,
                                            graph_mask)

    class TestMaxNodes(GNNCell):
        def construct(self, x, bg: BatchedGraph):
            return bg.max_nodes(x)

    ret = TestMaxNodes()(node_feat, *batched_graph_field.get_batched_graph()).asnumpy().tolist()
    expected = [
        [2, 4, 3, 4],
        [9, 7, 6, 8]
    ]
    first, second = np.array(ret), np.array(expected)
    assert np.sum(np.abs((first - second))) == 0


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_max_edges():
    """
    Feature:test max edges
    Description:test max edges
    Expectation:max_edges of edge_feat based on batched_graph_field
    """

    edge_feat = ms.Tensor([
        # graph 1:
        [1, 2, 3, 4],
        [2, 4, 1, 3],
        [1, 3, 2, 4],
        # graph 2:
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
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
    graph_mask = ms.Tensor([1, 1], ms.int32)
    batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx, edge_subgraph_idx,
                                            graph_mask)

    class TestMaxEdges(GNNCell):
        def construct(self, x, bg: BatchedGraph):
            return bg.max_edges(x)

    ret = TestMaxEdges()(edge_feat, *batched_graph_field.get_batched_graph()).asnumpy().tolist()
    expected = [
        [2, 4, 3, 4],
        [9, 7, 6, 8]
    ]
    first, second = np.array(ret), np.array(expected)
    assert np.sum(np.abs((first - second))) == 0


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_avg_nodes():
    """
    Feature:test avg nodes
    Description:test avg nodes
    Expectation:avg_nodes of node_feat based on batched_graph_field
    """

    node_feat = ms.Tensor([
        # graph 1:
        [1, 2, 3, 4],
        [2, 4, 1, 3],
        [1, 3, 2, 4],
        # graph 2:
        [9, 7, 5, 8],
        [8, 7, 6, 5],
        [8, 6, 4, 6],
        [1, 2, 1, 1],
    ], ms.float32)

    n_nodes = 7
    n_edges = 8
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
    graph_mask = ms.Tensor([1, 1], ms.int32)
    batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx, edge_subgraph_idx,
                                            graph_mask)

    class TestAvgNodes(GNNCell):
        def construct(self, x, bg: BatchedGraph):
            return bg.avg_nodes(x)

    ret = TestAvgNodes()(node_feat, *batched_graph_field.get_batched_graph()).asnumpy().tolist()
    expected = [
        [1.3333333730697632, 3.0, 2.0, 3.6666667461395264],
        [6.5, 5.5, 4.0, 5.0]
    ]
    first, second = np.array(ret), np.array(expected)
    assert np.sum(np.abs((first - second))) == 0


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_avg_edges():
    """
    Feature:test avg edges
    Description:test avg edges
    Expectation:avg_edges of edge_feat based on batched_graph_field
    """

    edge_feat = ms.Tensor([
        # graph 1:
        [1, 2, 3, 4],
        [2, 4, 1, 3],
        [1, 3, 2, 4],
        # graph 2:
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
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
    graph_mask = ms.Tensor([1, 1], ms.int32)
    batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx, edge_subgraph_idx,
                                            graph_mask)

    class TestAvgEdges(GNNCell):
        def construct(self, x, bg: BatchedGraph):
            return bg.avg_edges(x)

    ret = TestAvgEdges()(edge_feat, *batched_graph_field.get_batched_graph()).asnumpy().tolist()
    expected = [
        [1.3333333730697632, 3.0, 2.0, 3.6666667461395264],
        [5.800000190734863, 4.800000190734863, 3.799999952316284, 4.599999904632568]
    ]
    first, second = np.array(ret), np.array(expected)
    assert np.sum(np.abs((first - second))) == 0


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_softmax_nodes():
    """
    Feature:test softmax nodes
    Description:test softmax nodes
    Expectation:softmax_nodes of node_feat based on batched_graph_field
    """

    node_feat = ms.Tensor([
        # graph 1:
        [1, 2, 3, 4],
        [2, 4, 1, 3],
        [1, 3, 2, 4],
        # graph 2:
        [9, 7, 5, 8],
        [8, 7, 6, 5],
        [8, 6, 4, 6],
        [1, 2, 1, 1],
    ], ms.float32)

    n_nodes = 7
    n_edges = 8
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
    graph_mask = ms.Tensor([1, 1], ms.int32)
    batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx, edge_subgraph_idx,
                                            graph_mask)

    class TestSoftmaxNodes(GNNCell):
        def construct(self, x, bg: BatchedGraph):
            return bg.softmax_nodes(x)

    ret = TestSoftmaxNodes()(node_feat, *batched_graph_field.get_batched_graph()).asnumpy().tolist()
    expected = [
        [0.21194155514240265, 0.09003058075904846, 0.6652409434318542, 0.42231881618499756],
        [0.5761169195175171, 0.6652409434318542, 0.09003057330846786, 0.15536241233348846],
        [0.21194155514240265, 0.24472849071025848, 0.2447284758090973, 0.42231881618499756],
        [0.5760055780410767, 0.4211205244064331, 0.24363641440868378, 0.843146026134491],
        [0.21190060675144196, 0.4211205244064331, 0.6622724533081055, 0.04197777062654495],
        [0.21190060675144196, 0.15492157638072968, 0.08962882310152054, 0.1141074076294899],
        [0.00019322833395563066, 0.002837487729266286, 0.004462356213480234, 0.0007688496261835098]
    ]
    delta = 1e-5
    first, second = np.array(ret), np.array(expected)
    assert np.max(np.abs((first - second))) < delta


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_softmax_edges():
    """
    Feature:test softmax edges
    Description:test softmax edges
    Expectation:softmax_edges of edge_feat based on batched_graph_field
    """

    edge_feat = ms.Tensor([
        # graph 1:
        [1, 2, 3, 4],
        [2, 4, 1, 3],
        [1, 3, 2, 4],
        # graph 2:
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
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
    graph_mask = ms.Tensor([1, 1], ms.int32)
    batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx, edge_subgraph_idx,
                                            graph_mask)

    class TestSoftmaxEdges(GNNCell):
        def construct(self, x, bg: BatchedGraph):
            return bg.softmax_edges(x)

    ret = TestSoftmaxEdges()(edge_feat, *batched_graph_field.get_batched_graph()).asnumpy().tolist()
    expected = [
        [0.21194155514240265, 0.09003058075904846, 0.6652409434318542, 0.42231881618499756],
        [0.5761169195175171, 0.6652409434318542, 0.09003057330846786, 0.15536241233348846],
        [0.21194155514240265, 0.24472849071025848, 0.2447284758090973, 0.42231881618499756],
        [0.5751843452453613, 0.4199289381504059, 0.23585951328277588, 0.838383138179779],
        [0.2115984857082367, 0.4199289381504059, 0.641132652759552, 0.04174063727259636],
        [0.2115984857082367, 0.15448322892189026, 0.08676785975694656, 0.11346282064914703],
        [0.00019295283709652722, 0.0028294590301811695, 0.004319917410612106, 0.0007645064033567905],
        [0.0014257393777370453, 0.0028294590301811695, 0.031920112669467926, 0.005648980848491192]
    ]
    delta = 1e-5
    first, second = np.array(ret), np.array(expected)
    assert np.max(np.abs((first - second))) < delta


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_broadcast_nodes():
    """
    Feature:test broadcast nodes
    Description:test broadcast nodes
    Expectation:broadcast_nodes of node_feat based on batched_graph_field
    """

    node_feat = ms.Tensor([
        # graph 1:
        [1, 2, 3, 4],
        [2, 4, 1, 3],
        [1, 3, 2, 4],
        # graph 2:
        [9, 7, 5, 8],
        [8, 7, 6, 5],
        [8, 6, 4, 6],
        [1, 2, 1, 1],
    ], ms.float32)

    n_nodes = 7
    n_edges = 8
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
    graph_mask = ms.Tensor([1, 1], ms.int32)
    batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx, edge_subgraph_idx,
                                            graph_mask)

    class TestBroadCastNodes(GNNCell):
        def construct(self, x, bg: BatchedGraph):
            ret = bg.max_nodes(x)
            return bg.broadcast_nodes(ret)

    ret = TestBroadCastNodes()(node_feat, *batched_graph_field.get_batched_graph()).asnumpy().tolist()
    expected = [
        [2, 4, 3, 4],
        [2, 4, 3, 4],
        [2, 4, 3, 4],
        [9, 7, 6, 8],
        [9, 7, 6, 8],
        [9, 7, 6, 8],
        [9, 7, 6, 8],
    ]
    first, second = np.array(ret), np.array(expected)
    assert np.sum(np.abs((first - second))) == 0


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_broadcast_edges():
    """
    Feature:test broadcast edges
    Description:test broadcast edges
    Expectation:broadcast_edges of edge_feat based on batched_graph_field
    """

    edge_feat = ms.Tensor([
        # graph 1:
        [1, 2, 3, 4],
        [2, 4, 1, 3],
        [1, 3, 2, 4],
        # graph 2:
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
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
    graph_mask = ms.Tensor([1, 1], ms.int32)
    batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx, edge_subgraph_idx,
                                            graph_mask)

    class TestBroadCastEdges(GNNCell):
        def construct(self, x, bg: BatchedGraph):
            ret = bg.max_edges(x)
            return bg.broadcast_edges(ret)

    ret = TestBroadCastEdges()(edge_feat, *batched_graph_field.get_batched_graph()).asnumpy().tolist()
    expected = [
        [2, 4, 3, 4],
        [2, 4, 3, 4],
        [2, 4, 3, 4],
        [9, 7, 6, 8],
        [9, 7, 6, 8],
        [9, 7, 6, 8],
        [9, 7, 6, 8],
        [9, 7, 6, 8],
    ]
    first, second = np.array(ret), np.array(expected)
    assert np.sum(np.abs((first - second))) == 0


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
        # graph 1:
        [1, 2, 3, 4],
        [2, 4, 1, 3],
        [1, 3, 2, 4],
        # graph 2:
        [9, 7, 5, 8],
        [8, 7, 6, 5],
        [8, 6, 4, 6],
        [1, 2, 1, 1],
    ], ms.float32)

    n_nodes = 7
    n_edges = 8
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
    graph_mask = ms.Tensor([1, 1], ms.int32)
    batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx, edge_subgraph_idx,
                                            graph_mask)

    class TestTopkNodes(GNNCell):
        def construct(self, x, bg: BatchedGraph):
            return bg.topk_nodes(x, 2)

    output, indices = TestTopkNodes()(node_feat, *batched_graph_field.get_batched_graph())
    output = output.asnumpy().tolist()
    indices = indices.asnumpy().tolist()
    output_expected = [
        [[2, 4, 3, 4],
         [1, 3, 2, 4]],
        [[9, 7, 6, 8],
         [8, 7, 5, 6]],
    ]
    indices_expected = [
        [[1, 1, 0, 0],
         [0, 2, 2, 2]],
        [[3, 3, 4, 3],
         [4, 4, 3, 5]]
    ]
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
        # graph 1:
        [1, 2, 3, 4],
        [2, 4, 1, 3],
        [1, 3, 2, 4],
        # graph 2:
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
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
    graph_mask = ms.Tensor([1, 1], ms.int32)
    batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx, edge_subgraph_idx,
                                            graph_mask)

    class TestTopkEdges(GNNCell):
        def construct(self, x, bg: BatchedGraph):
            return bg.topk_edges(x, 2)

    output, indices = TestTopkEdges()(edge_feat, *batched_graph_field.get_batched_graph())
    output = output.asnumpy().tolist()
    indices = indices.asnumpy().tolist()
    output_expected = [
        [[2, 4, 3, 4],
         [1, 3, 2, 4]],
        [[9, 7, 6, 8],
         [8, 7, 5, 6]],
    ]
    indices_expected = [
        [[1, 1, 0, 0],
         [0, 2, 2, 2]],
        [[3, 3, 4, 3],
         [4, 4, 3, 5]]
    ]
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
        # graph 1:
        [1, 2, 3, 4],
        [2, 4, 1, 3],
        [1, 3, 2, 4],
        # graph 2:
        [9, 7, 5, 8],
        [8, 7, 6, 5],
        [8, 6, 4, 6],
        [1, 2, 1, 1],
    ], ms.float32)

    n_nodes = 7
    n_edges = 8
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
    graph_mask = ms.Tensor([1, 1], ms.int32)
    batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx, edge_subgraph_idx,
                                            graph_mask)

    class TestTopkNodes(GNNCell):
        def construct(self, x, bg: BatchedGraph):
            return bg.topk_nodes(x, 2, 1)

    output, indices = TestTopkNodes()(node_feat, *batched_graph_field.get_batched_graph())
    output = output.asnumpy().tolist()
    indices = indices.asnumpy().tolist()
    output_expected = [
        [[2, 4, 1, 3],
         [1, 3, 2, 4]],
        [[9, 7, 5, 8],
         [8, 7, 6, 5]],
    ]
    indices_expected = [
        [1, 2],
        [3, 4]
    ]
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
        # graph 1:
        [1, 2, 3, 4],
        [2, 4, 1, 3],
        [1, 3, 2, 4],
        # graph 2:
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
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
    graph_mask = ms.Tensor([1, 1], ms.int32)
    batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx, edge_subgraph_idx,
                                            graph_mask)

    class TestTopkEdges(GNNCell):
        def construct(self, x, bg: BatchedGraph):
            return bg.topk_edges(x, 2, 1)

    output, indices = TestTopkEdges()(edge_feat, *batched_graph_field.get_batched_graph())
    output = output.asnumpy().tolist()
    indices = indices.asnumpy().tolist()
    output_expected = [
        [[2, 4, 1, 3],
         [1, 3, 2, 4]],
        [[9, 7, 5, 8],
         [8, 7, 6, 5]],
    ]
    indices_expected = [
        [1, 2],
        [3, 4]
    ]
    first, second = np.array(output), np.array(output_expected)
    assert np.sum(np.abs((first - second))) == 0
    first, second = np.array(indices), np.array(indices_expected)
    assert np.sum(np.abs((first - second))) == 0
