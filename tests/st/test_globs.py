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
""" test globs """
import mindspore as ms
import mindspore.context as context
import mindspore.numpy as np
import pytest
import numpy

from mindspore_gl.nn import SumPooling, MaxPooling, AvgPooling, SortPooling, \
    GlobalAttentionPooling, WeightAndSum, Set2Set, SAGPooling
from mindspore_gl import BatchedGraphField

context.set_context(device_target="GPU", mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sumpooling():
    """
    Feature:test sumpooling
    Description:test sumpooling
    Expectation:sumpooling of node_feat based on batched_graph_field
    """

    n_nodes = 7
    n_edges = 8
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
    graph_mask = ms.Tensor([1, 1], ms.int32)
    batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx, edge_subgraph_idx,
                                            graph_mask)

    node_feat = ms.Tensor(
        [
            [0.8948, 0.0699, 0.9137, 0.7567, 0.3637],
            [0.8137, 0.8938, 0.8377, 0.4249, 0.6118],
            [0.5197, 0.9030, 0.6825, 0.5725, 0.4755],
            [0.2053, 0.2426, 0.4111, 0.9028, 0.5658],
            [0.5278, 0.6365, 0.9990, 0.2351, 0.8945],
            [0.3134, 0.0580, 0.4349, 0.7949, 0.3891],
            [0.0142, 0.2709, 0.3330, 0.8521, 0.6925],
        ]
        , ms.float32)

    net = SumPooling()
    ret = net(node_feat, *batched_graph_field.get_batched_graph()).asnumpy().tolist()
    expected = [[2.2282, 1.8666999, 2.4338999, 1.7541, 1.451], [1.0606999, 1.208, 2.178, 2.7849002, 2.5419]]

    delta = 1e-5
    first, second = np.array(ret), np.array(expected)
    assert np.max(np.abs((first - second))) < delta


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maxpooling():
    """
    Feature:test maxpooling
    Description:test maxpooling
    Expectation:maxpooling of node_feat based on batched_graph_field
    """

    n_nodes = 7
    n_edges = 8
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
    graph_mask = ms.Tensor([1, 1], ms.int32)
    batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx, edge_subgraph_idx,
                                            graph_mask)

    node_feat = ms.Tensor(
        [
            [0.8948, 0.0699, 0.9137, 0.7567, 0.3637],
            [0.8137, 0.8938, 0.8377, 0.4249, 0.6118],
            [0.5197, 0.9030, 0.6825, 0.5725, 0.4755],
            [0.2053, 0.2426, 0.4111, 0.9028, 0.5658],
            [0.5278, 0.6365, 0.9990, 0.2351, 0.8945],
            [0.3134, 0.0580, 0.4349, 0.7949, 0.3891],
            [0.0142, 0.2709, 0.3330, 0.8521, 0.6925],
        ]
        , ms.float32)

    net = MaxPooling()
    ret = net(node_feat, *batched_graph_field.get_batched_graph()).asnumpy().tolist()
    expected = [[0.8948, 0.903, 0.9137, 0.7567, 0.6118], [0.5278, 0.6365, 0.999, 0.9028, 0.8945]]

    delta = 1e-5
    first, second = np.array(ret), np.array(expected)
    assert np.max(np.abs((first - second))) < delta


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_avgpooling():
    """
    Feature:test avgpooling
    Description:test avgpooling
    Expectation:avgpooling of node_feat based on batched_graph_field
    """

    n_nodes = 7
    n_edges = 8
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
    graph_mask = ms.Tensor([1, 1], ms.int32)
    batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx, edge_subgraph_idx,
                                            graph_mask)

    node_feat = ms.Tensor(
        [
            [0.8948, 0.0699, 0.9137, 0.7567, 0.3637],
            [0.8137, 0.8938, 0.8377, 0.4249, 0.6118],
            [0.5197, 0.9030, 0.6825, 0.5725, 0.4755],
            [0.2053, 0.2426, 0.4111, 0.9028, 0.5658],
            [0.5278, 0.6365, 0.9990, 0.2351, 0.8945],
            [0.3134, 0.0580, 0.4349, 0.7949, 0.3891],
            [0.0142, 0.2709, 0.3330, 0.8521, 0.6925],
        ]
        , ms.float32)

    net = AvgPooling()
    ret = net(node_feat, *batched_graph_field.get_batched_graph()).asnumpy().tolist()
    expected = [[0.7427333, 0.6222333, 0.81129996, 0.5847, 0.483666],
                [0.265174975, 0.302, 0.5445, 0.69622505, 0.635475]]

    delta = 1e-5
    first, second = np.array(ret), np.array(expected)
    assert np.max(np.abs((first - second))) < delta


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sortpooling():
    """
    Feature:test sortpooling
    Description:test sortpooling
    Expectation:sortpooling of node_feat based on batched_graph_field
    """

    n_nodes = 7
    n_edges = 8
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
    graph_mask = ms.Tensor([1, 1], ms.int32)
    batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx, edge_subgraph_idx,
                                            graph_mask)

    node_feat = ms.Tensor(
        [
            [0.8948, 0.0699, 0.9137, 0.7567, 0.3637],
            [0.8137, 0.8938, 0.8377, 0.4249, 0.6118],
            [0.5197, 0.9030, 0.6825, 0.5725, 0.4755],
            [0.2053, 0.2426, 0.4111, 0.9028, 0.5658],
            [0.5278, 0.6365, 0.9990, 0.2351, 0.8945],
            [0.3134, 0.0580, 0.4349, 0.7949, 0.3891],
            [0.0142, 0.2709, 0.3330, 0.8521, 0.6925],
        ]
        , ms.float32)

    net = SortPooling(k=2)
    ret = net(node_feat, *batched_graph_field.get_batched_graph()).asnumpy().tolist()
    expected = [[0.0699, 0.3637, 0.7567, 0.8948, 0.9137, 0.4755, 0.5197, 0.5725, 0.6825, 0.903],
                [0.2351, 0.5278, 0.6365, 0.8945, 0.999, 0.2053, 0.2426, 0.4111, 0.5658, 0.9028]]

    delta = 1e-5
    first, second = np.array(ret), np.array(expected)
    assert np.max(np.abs((first - second))) < delta


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_globalattentionpooling():
    """
    Feature:test globalattentionpooling
    Description:test globalattentionpooling
    Expectation:globalattentionpooling of node_feat based on batched_graph_field
    """

    n_nodes = 7
    n_edges = 8
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
    graph_mask = ms.Tensor([1, 1], ms.int32)
    batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx, edge_subgraph_idx,
                                            graph_mask)

    node_feat = ms.Tensor(
        [
            [0.8948, 0.0699, 0.9137, 0.7567, 0.3637],
            [0.8137, 0.8938, 0.8377, 0.4249, 0.6118],
            [0.5197, 0.9030, 0.6825, 0.5725, 0.4755],
            [0.2053, 0.2426, 0.4111, 0.9028, 0.5658],
            [0.5278, 0.6365, 0.9990, 0.2351, 0.8945],
            [0.3134, 0.0580, 0.4349, 0.7949, 0.3891],
            [0.0142, 0.2709, 0.3330, 0.8521, 0.6925],
        ]
        , ms.float32)

    gate_nn = ms.nn.Dense(5, 1)
    net = GlobalAttentionPooling(gate_nn)
    ret = net(node_feat, *batched_graph_field.get_batched_graph()).asnumpy().tolist()
    expected = [[0.7427342, 0.6216653, 0.8113234, 0.5849572, 0.4834695],
                [0.2650575, 0.301947, 0.54436606, 0.69635034, 0.63545763]]

    delta = 0.05
    first, second = np.array(ret), np.array(expected)
    assert np.max(np.abs((first - second))) < delta


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_weightandsum():
    """
    Feature:test weightandsum
    Description:test weightandsum
    Expectation:weightandsum of node_feat based on batched_graph_field
    """

    n_nodes = 7
    n_edges = 8
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
    graph_mask = ms.Tensor([1, 1], ms.int32)
    batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx, edge_subgraph_idx,
                                            graph_mask)

    node_feat = ms.Tensor(
        [
            [0.8948, 0.0699, 0.9137, 0.7567, 0.3637],
            [0.8137, 0.8938, 0.8377, 0.4249, 0.6118],
            [0.5197, 0.9030, 0.6825, 0.5725, 0.4755],
            [0.2053, 0.2426, 0.4111, 0.9028, 0.5658],
            [0.5278, 0.6365, 0.9990, 0.2351, 0.8945],
            [0.3134, 0.0580, 0.4349, 0.7949, 0.3891],
            [0.0142, 0.2709, 0.3330, 0.8521, 0.6925],
        ]
        , ms.float32)

    net = WeightAndSum(5)
    ret = net(node_feat, *batched_graph_field.get_batched_graph()).asnumpy().tolist()
    expected = [[1.1191536, 0.9370661, 1.222404, 0.8810187, 0.7286275],
                [0.5323895, 0.6062569, 1.0929173, 1.3955828, 1.2749455]]

    delta = 0.05
    first, second = np.array(ret), np.array(expected)
    assert np.max(np.abs((first - second))) < delta


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_set2set():
    """
    Feature:test set2set
    Description:test set2set
    Expectation:set2set of node_feat based on batched_graph_field
    """

    n_nodes = 7
    n_edges = 8
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
    graph_mask = ms.Tensor([1, 1], ms.int32)
    batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx, edge_subgraph_idx,
                                            graph_mask)

    node_feat = ms.Tensor(
        [
            [0.8948, 0.0699, 0.9137, 0.7567, 0.3637],
            [0.8137, 0.8938, 0.8377, 0.4249, 0.6118],
            [0.5197, 0.9030, 0.6825, 0.5725, 0.4755],
            [0.2053, 0.2426, 0.4111, 0.9028, 0.5658],
            [0.5278, 0.6365, 0.9990, 0.2351, 0.8945],
            [0.3134, 0.0580, 0.4349, 0.7949, 0.3891],
            [0.0142, 0.2709, 0.3330, 0.8521, 0.6925],
        ]
        , ms.float32)

    net = Set2Set(5, 2, 1)
    ret = net(node_feat, *batched_graph_field.get_batched_graph()).asnumpy().tolist()
    expected = [[0, 0.1, 0.2, 0, 0, 0.7, 0.6, 0.8, 0.6, 0.5],
                [0, 0, 0.1, 0, -0.1, 0.2, 0.3, 0.6, 0.6, 0.6]]

    delta = 0.6
    first, second = np.array(ret), np.array(expected)
    assert np.max(np.abs((first - second))) < delta


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sagpooling():
    """
    Features: SAGPooling
    Description: Test Sagpooling.
    Expectation: The output is as expected.
    """
    node_feat = ms.Tensor([
        [1, 2, 3, 4],
        [2, 4, 1, 3],
        [1, 3, 2, 4],
        [9, 7, 5, 8],
        [8, 7, 6, 5],
        [8, 6, 4, 6],
        [1, 2, 1, 1]
    ], ms.float32)
    n_nodes = 7
    n_edges = 8
    src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
    dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
    ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
    edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
    graph_mask = ms.Tensor([0, 1], ms.int32)
    batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx, edge_subgraph_idx,
                                            graph_mask)
    weight_gcn = np.array([-0.43797874, -0.24720865, -0.32895839, 0.29777485])
    weight_gcn = ms.Tensor(weight_gcn, ms.float32).view((1, 4))
    bias_gcn = np.array(-0.23705053)
    bias_gcn = ms.Tensor(bias_gcn, ms.float32).view(1)
    weight_gcn2 = np.array([0.38274187, -0.24117965, -0.35172099, -0.12423509])
    weight_gcn2 = ms.Tensor(weight_gcn2, ms.float32).view((1, 4))
    expect_feature = numpy.array([[[-0.94450444, -2.83351326, -1.88900888, -3.77801776],
                                   [-0.98751843, -1.97503686, -2.96255541, -3.95007372]],
                                  [[-0.99995297, -1.99990594, -0.99995297, -0.99995297],
                                   [-7.999824, -5.999868, -3.999912, -5.999868]]])
    expect_src = numpy.array([0., 3.])
    expect_dst = numpy.array([1., 2.])
    expect_ver_subgraph = numpy.array([0., 0., 1., 1.])
    expect_edge_subgraph = numpy.array([0., 1.])
    expect_perm = numpy.array([2., 0., 6., 5.])
    expect_score = numpy.array([[[-0.94450444], [-0.98751843]],
                                [[-0.99995297], [-0.999978]]])
    net = SAGPooling(4)
    net.gnn.fc1.weight.set_data(weight_gcn)
    net.gnn.fc2.weight.set_data(weight_gcn2)
    net.gnn.bias.set_data(bias_gcn)
    feature, src, dst, ver_subgraph, edge_subgraph, perm, perm_score = net(node_feat, None, 2,
                                                                           *batched_graph_field.get_batched_graph())
    assert numpy.allclose(feature.asnumpy(), expect_feature)
    assert numpy.allclose(src.asnumpy(), expect_src)
    assert numpy.allclose(dst.asnumpy(), expect_dst)
    assert numpy.allclose(ver_subgraph.asnumpy(), expect_ver_subgraph)
    assert numpy.allclose(edge_subgraph.asnumpy(), expect_edge_subgraph)
    assert numpy.allclose(perm.asnumpy(), expect_perm)
    assert numpy.allclose(perm_score.asnumpy(), expect_score)
