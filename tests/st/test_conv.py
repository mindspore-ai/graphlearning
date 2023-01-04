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
"""Test Conv Layers."""
import pytest
import numpy as np
import mindspore as ms
from mindspore import nn
import mindspore.context as context
from mindspore.numpy import ones
from mindspore_gl import GraphField, BatchedGraphField
from mindspore_gl.graph import norm
from mindspore_gl.nn import AGNNConv
from mindspore_gl.nn import APPNPConv
from mindspore_gl.nn import CFConv
from mindspore_gl.nn import ChebConv
from mindspore_gl.nn import DOTGATConv
from mindspore_gl.nn import EDGEConv
from mindspore_gl.nn import EGConv
from mindspore_gl.nn import GATConv
from mindspore_gl.nn import GatedGraphConv
from mindspore_gl.nn import GATv2Conv
from mindspore_gl.nn import GCNConv2
from mindspore_gl.nn import STConv
from mindspore_gl.nn import TAGConv
from mindspore_gl.nn import SAGEConv
from mindspore_gl.nn import NNConv
from mindspore_gl.nn import GMMConv
from mindspore_gl.nn import MeanConv
from mindspore_gl.nn import GINConv
from mindspore_gl.nn import GCNConv
from mindspore_gl.nn import SGConv

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
edge_feat_size = 7
edge_feat = ones([n_edges, 4], ms.float32)
src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
in_degree = np.zeros(shape=n_nodes, dtype=np.int)
out_degree = np.zeros(shape=n_nodes, dtype=np.int)
for r in src_idx:
    out_degree[r] += 1
for r in dst_idx:
    in_degree[r] += 1
in_degree = ms.Tensor(in_degree, ms.int32)
out_degree = ms.Tensor(out_degree, ms.int32)
ver_subgraph_idx = ms.Tensor([0, 0, 0, 0, 0, 0, 0], ms.int32)
edge_subgraph_idx = ms.Tensor([0, 0, 0, 0, 0, 0, 0, 0], ms.int32)
graph_mask = ms.Tensor([1], ms.int32)
batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges,
                                        ver_subgraph_idx, edge_subgraph_idx, graph_mask)
graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)


# test conv api
@pytest.mark.lebel0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_agnnconv():
    """
    Features: AGNNConv.
    Description: Test AGNNConv.
    Expectation: The output is as expected.
    """
    net = AGNNConv(0.5)
    expect_output = np.array([[1, 3, 2, 4], [0.99999994, 2.501635, 2.4983644, 3.9999998],
                              [0, 0, 0, 0], [8, 7, 6, 5],
                              [5.1674695, 4.381411, 2.7860584, 3.976764], [9, 7, 5, 8],
                              [8, 6, 4, 6]])
    output = net(node_feat, *graph_field.get_graph())
    assert np.allclose(output.asnumpy(), expect_output)


@pytest.mark.lebel0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_appnpconv():
    """
    Features: APPNPConv.
    Description: Test APPNPConv.
    Expectation: The output is as expected.
    """
    in_deg = ms.Tensor([1, 2, 0, 1, 2, 1, 1], ms.int32)
    out_deg = ms.Tensor([1, 0, 2, 1, 1, 2, 1], ms.int32)
    net = APPNPConv(k=3, alpha=0.5, edge_drop=1.0)

    output = net(node_feat, in_deg, out_deg, *graph_field.get_graph())
    expect_output = np.array([[0.75, 1.75, 2., 3.],
                              [1.625, 3.625, 2., 4.],
                              [0.5, 1.5, 1., 2.],
                              [7.979315, 6.4123735, 4.6686554, 6.5329313],
                              [9.34099, 7.9249363, 6.0088835, 6.726713],
                              [6.65349, 5.1749363, 3.5713832, 5.164213],
                              [3.795495, 3.5562181, 2.3169417, 3.0196066]])
    assert np.allclose(output.asnumpy(), expect_output)


@pytest.mark.lebel0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cfconv():
    """
    Features: CFConv.
    Description: Test CFConv.
    Expectation: The output is as expected.
    """
    edge_embedding_weight_1 = ms.Tensor(np.ones((8, 4)), ms.float32)
    edge_embedding_weight_2 = ms.Tensor(np.ones((8, 8)), ms.float32)
    node_embedding_weight = ms.Tensor(np.ones((8, 4)), ms.float32)
    out_embedding_weight = ms.Tensor(np.ones((4, 8)), ms.float32)

    net = CFConv(4, 4, 8, 4)
    net.edge_embedding_layer[0].weight.set_data(edge_embedding_weight_1)
    net.edge_embedding_layer[2].weight.set_data(edge_embedding_weight_2)
    net.node_embedding_layer.weight.set_data(node_embedding_weight)
    net.out_embedding_layer[0].weight.set_data(out_embedding_weight)
    output = net(node_feat, edge_feat, *graph_field.get_graph())
    expect_output = np.array([[2071.8567, 2071.8567, 2071.8567, 2071.8567],
                              [4144.4062, 4144.4062, 4144.4062, 4144.4062],
                              [0., 0., 0., 0.],
                              [5387.936, 5387.936, 5387.936, 5387.936],
                              [6009.701, 6009.701, 6009.701, 6009.701],
                              [6009.701, 6009.701, 6009.701, 6009.701],
                              [4973.4263, 4973.4263, 4973.4263, 4973.4263]])
    assert np.allclose(output.asnumpy(), expect_output)


@pytest.mark.lebel0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_chebconv():
    """
    Features: ChebConv.
    Description: Test ChebConv.
    Expectation: The output is as expected.
    """
    lins_weight_0 = ms.Tensor(np.ones((4, 4)), ms.float32)
    lins_weight_1 = ms.Tensor(np.ones((4, 4)), ms.float32)
    lins_weight_2 = ms.Tensor(np.ones((4, 4)), ms.float32)

    net = ChebConv(in_channels=4, out_channels=4, k=3)
    net.lins[0].weight.set_data(lins_weight_0)
    net.lins[1].weight.set_data(lins_weight_1)
    net.lins[2].weight.set_data(lins_weight_2)
    output = net(node_feat, edge_feat, *graph_field.get_graph())
    expect_output = np.array([[10., 10., 10., 10.],
                              [40., 40., 40., 40.],
                              [0., 0., 0., 0.],
                              [84., 84., 84., 84.],
                              [135., 135., 135., 135.],
                              [81., 81., 81., 81.],
                              [82., 82., 82., 82.]])
    assert np.allclose(output.asnumpy(), expect_output)


@pytest.mark.lebel0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dotgatconv():
    """
    Features: DOTGATConv.
    Description: Test DOTGATConv.
    Expectation: The output is as expected.
    """
    dense_weight = ms.Tensor(np.ones((8, 4)), ms.float32)

    net = DOTGATConv(4, 4, 2)
    net.dense.weight.set_data(dense_weight)
    output = net(node_feat, *graph_field.get_graph())
    expect_output = np.array([[[10., 10., 10., 10.], [10., 10., 10., 10.]],
                              [[10., 10., 10., 10.], [10., 10., 10., 10.]],
                              [[0., 0., 0., 0.], [0., 0., 0., 0.]],
                              [[26., 26., 26., 26.], [26., 26., 26., 26.]],
                              [[20.724138, 20.724138, 20.724138, 20.724138],
                               [20.724138, 20.724138, 20.724138, 20.724138]],
                              [[29., 29., 29., 29.], [29., 29., 29., 29.]],
                              [[24., 24., 24., 24.], [24., 24., 24., 24.]]])
    assert np.allclose(output.asnumpy(), expect_output)


@pytest.mark.lebel0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_edgeconv():
    """
    Features: EDGEConv.
    Description: Test EDGEConv.
    Expectation: The output is as expected.
    """
    theta_weight = ms.Tensor(np.ones((4, 4)), ms.float32)
    phi_weight = ms.Tensor(np.ones((4, 4)), ms.float32)

    net = EDGEConv(4, 4, batch_norm=True)
    net.theta.weight.set_data(theta_weight)
    net.phi.weight.set_data(phi_weight)
    output = net(node_feat, *graph_field.get_graph())
    expect_output = np.array([[9.99995, 9.99995, 9.99995, 9.99995],
                              [9.99995, 9.99995, 9.99995, 9.99995],
                              [0., 0., 0., 0.],
                              [25.99987, 25.99987, 25.99987, 25.99987],
                              [23.99988, 23.99988, 23.99988, 23.99988],
                              [28.999855, 28.999855, 28.999855, 28.999855],
                              [23.99988, 23.99988, 23.99988, 23.99988]])
    assert np.allclose(output.asnumpy(), expect_output)


@pytest.mark.lebel0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_egconv():
    """
    Features: EGConv.
    Description: Test EGConv.
    Expectation: The output is as expected.
    """
    basis_fc_weight = ms.Tensor(np.ones((6, 4)), ms.float32)
    combine_fc_weight = ms.Tensor(np.ones((9, 4)), ms.float32)

    net = EGConv(in_feat_size=4, out_feat_size=6, aggregators=['sum'], num_heads=3, num_bases=3)
    net.basis_fc.weight.set_data(basis_fc_weight)
    net.combine_fc.weight.set_data(combine_fc_weight)
    output = net(node_feat, *graph_field.get_graph())
    expect_output = np.array([[300., 300., 300., 300., 300., 300.],
                              [600., 600., 600., 600., 600., 600.],
                              [0., 0., 0., 0., 0., 0.],
                              [2262., 2262., 2262., 2262., 2262., 2262.],
                              [2262., 2262., 2262., 2262., 2262., 2262.],
                              [2088., 2088., 2088., 2088., 2088., 2088.],
                              [360., 360., 360., 360., 360., 360.]])
    assert np.allclose(output.asnumpy(), expect_output)


@pytest.mark.lebel0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gatconv():
    """
    Features: GATConv.
    Description: Test GATConv.
    Expectation: The output is as expected.
    """
    attn_s_data = ms.Tensor(np.ones((3, 2)), ms.float32)
    attn_d_data = ms.Tensor(np.ones((3, 2)), ms.float32)
    fc_weight = ms.Tensor(np.ones((6, 4)), ms.float32)

    net = GATConv(in_feat_size=4, out_size=2, num_attn_head=3, add_norm=True)
    net.fc.weight.set_data(fc_weight)
    net.attn_s.set_data(attn_s_data)
    net.attn_d.set_data(attn_d_data)
    output = net(node_feat, *graph_field.get_graph())
    expect_output = np.array([[10., 10., 10., 10., 10., 10.],
                              [10., 10., 10., 10., 10., 10.],
                              [0., 0., 0., 0., 0., 0.],
                              [26., 26., 26., 26., 26., 26.],
                              [16.283588, 16.283588, 16.283588, 16.283588, 16.283588, 16.283588],
                              [29., 29., 29., 29., 29., 29.],
                              [24., 24., 24., 24., 24., 24.]])
    assert np.allclose(output.asnumpy(), expect_output)


@pytest.mark.lebel0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gatedgraphconv():
    """
    Features: GatedGraphConv.
    Description: Test GatedGraphConv.
    Expectation: The output is as expected.
    """
    src_idx_list = [ms.Tensor([0, 2, 2], ms.int32), ms.Tensor([3, 4], ms.int32),
                    ms.Tensor([5, 5, 6], ms.int32)]
    dst_idx_list = [ms.Tensor([1, 0, 1], ms.int32), ms.Tensor([5, 3], ms.int32),
                    ms.Tensor([4, 6, 4], ms.int32)]
    n_edges_list = [3, 2, 3]

    dense_weight_0 = ms.Tensor(np.ones((4, 4)), ms.float32)
    dense_weight_1 = ms.Tensor(np.ones((4, 4)), ms.float32)
    dense_weight_2 = ms.Tensor(np.ones((4, 4)), ms.float32)
    gru_ih_weight = ms.Tensor(np.ones((12, 4)), ms.float32)
    gru_hh_weight = ms.Tensor(np.ones((12, 4)), ms.float32)
    gru_ih_bias = ms.Tensor(np.zeros((12,)), ms.float32)
    gru_hh_bias = ms.Tensor(np.zeros((12,)), ms.float32)

    net = GatedGraphConv(4, 4, 2, 3, True)
    net.cell_list[0].dense.weight.set_data(dense_weight_0)
    net.cell_list[1].dense.weight.set_data(dense_weight_1)
    net.cell_list[2].dense.weight.set_data(dense_weight_2)
    net.gru.weight_ih.set_data(gru_ih_weight)
    net.gru.weight_hh.set_data(gru_hh_weight)
    net.gru.bias_ih.set_data(gru_ih_bias)
    net.gru.bias_hh.set_data(gru_hh_bias)

    output = net(node_feat, src_idx_list, dst_idx_list, n_nodes, n_edges_list)
    expect_output = np.array([[1., 2., 3., 4.],
                              [2., 4., 1., 3.],
                              [1., 2.9998183, 1.9999092, 3.9997275],
                              [9., 7., 5., 8.],
                              [8., 7., 6., 5.],
                              [8., 6., 4., 6.],
                              [1., 2., 1., 1.]])
    assert np.allclose(output.asnumpy(), expect_output)


@pytest.mark.lebel0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gatv2conv():
    """
    Features: GATv2Conv.
    Description: Test GATv2Conv.
    Expectation: The output is as expected.
    """
    fc_s_weight = ms.Tensor(np.ones((6, 4)), ms.float32)
    fc_d_weight = ms.Tensor(np.ones((6, 4)), ms.float32)
    attn_data = ms.Tensor(np.ones((3, 2)), ms.float32)

    net = GATv2Conv(in_feat_size=4, out_size=2, num_attn_head=3, add_norm=True)
    net.fc_s.weight.set_data(fc_s_weight)
    net.fc_d.weight.set_data(fc_d_weight)
    net.attn.set_data(attn_data)
    output = net(node_feat, *graph_field.get_graph())
    expect_output = np.array([[10., 10., 10., 10., 10., 10.],
                              [10., 10., 10., 10., 10., 10.],
                              [0., 0., 0., 0., 0., 0.],
                              [26., 26., 26., 26., 26., 26.],
                              [16.283588, 16.283588, 16.283588, 16.283588, 16.283588, 16.283588],
                              [29., 29., 29., 29., 29., 29.],
                              [24., 24., 24., 24., 24., 24.]])
    assert np.allclose(output.asnumpy(), expect_output)


@pytest.mark.lebel0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gcnconv2():
    """
    Features: GCNConv2
    Description: Test GCNConv2.
    Expectation: The output is as expected.
    """
    weight_gcn = np.array([-0.21176505, -0.33282989, 0.32413161, 0.44888139])
    weight_gcn = ms.Tensor(weight_gcn, ms.float32).view((1, 4))
    bias_gcn = np.array(0.34598231)
    bias_gcn = ms.Tensor(bias_gcn, ms.float32).view(1)
    weight_gcn2 = np.array([-0.48440778, -0.30549908, 0.08788288, 0.25465935])
    weight_gcn2 = ms.Tensor(weight_gcn2, ms.float32).view((1, 4))
    expect_output = np.array([[1.76639652], [2.13106108], [0.13948047], [-3.51022506],
                              [-3.67287588], [-2.50677252], [-0.1081664]])

    net = GCNConv2(4, 1)
    net.fc1.weight.set_data(weight_gcn)
    net.fc2.weight.set_data(weight_gcn2)
    net.bias.set_data(bias_gcn)
    output = net(node_feat, *graph_field.get_graph())
    assert np.allclose(output.asnumpy(), expect_output)


@pytest.mark.lebel0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tagconv():
    """
    Features: TAGConv
    Description: Test TAGConv.
    Expectation: The output is as expected.
    """
    weight_tag = np.array([[0.7443847, 0.8096786, -0.55130184, 0.5126086, -0.4128897, 0.01070621, 0.6337174,
                            0.48398262, -0.25557196, -0.82113224, 0.25976783, 0.63135546, -0.12402994,
                            -0.13683635, 0.03259707, -0.7975022]])
    weight_tag = ms.Tensor(weight_tag, ms.float32)
    bias_tag = ms.Tensor([0.], ms.float32)
    expect_output = np.array([[4.756146], [9.72471], [4.121251], [7.5421596],
                              [3.5536532], [7.949831], [-1.3805635]])

    tagconv = TAGConv(in_feat_size=4, out_feat_size=1, activation=None, num_hops=3)
    tagconv.dense.weight.set_data(weight_tag)
    tagconv.dense.bias.set_data(bias_tag)
    output = tagconv(node_feat, in_degree, out_degree, *graph_field.get_graph())
    assert np.allclose(output.asnumpy(), expect_output)


@pytest.mark.lebel0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sageconv():
    """
    Features: SAGEConv
    Description: Test SAGEConv.
    Expectation: The output is as expected.
    """
    dense_neigh_weight = ms.Tensor([[0.12024725, -0.94418734, -0.68363243, 0.6015784],
                                    [0.37799108, 1.2374284, 0.98415923, -0.6093902]],
                                   ms.float32)
    fc_pool_weight = ms.Tensor([[-0.00624315, -0.0216758, 0.00936903, 0.01202865],
                                [-0.00362654, -0.00729788, 0.01195601, 0.00978783],
                                [0.00329617, 0.00036223, 0.01229328, -0.01320718],
                                [-0.00241942, 0.00531684, 0.02327169, -0.00389408]],
                               ms.float32)
    fc_pool_bias = ms.Tensor([0., 0., 0., 0.], ms.float32)
    dense_self_weight = ms.Tensor([[0.9841772, -1.251965, 1.0914761, 0.9614603],
                                   [0.2935081, 0.8534617, -0.49380463, 0.27783674]],
                                  ms.float32)
    expect_output = np.array([[5.591838, 1.6497049], [0.92349607, 4.3828573], [3.2570753, 2.9776306],
                              [13.262579, 8.371627], [10.481319, 6.7433414], [10.505516, 7.166567],
                              [0.55080634, 1.7760953]])
    sageconv = SAGEConv(in_feat_size=4, out_feat_size=2, activation=nn.ReLU())
    sageconv.dense_neigh.weight.set_data(dense_neigh_weight)
    sageconv.fc_pool.weight.set_data(fc_pool_weight)
    sageconv.fc_pool.bias.set_data(fc_pool_bias)
    sageconv.dense_self.weight.set_data(dense_self_weight)
    edge_weight = ones((n_edges, 1), ms.float32)
    output = sageconv(node_feat, edge_weight, *graph_field.get_graph())
    assert np.allclose(output.asnumpy(), expect_output)


@pytest.mark.lebel0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_nnconv():
    """
    Features: NNConv
    Description: Test NNConv.
    Expectation: The output is as expected.
    """
    nn_weight = ms.Tensor([[-0.00425986, -0.00037071, -0.01033616, 0.01928994, 0.01009436, -0.00024363, -0.00383958],
                           [-0.0150689, -0.00871064, 0.00025874, 0.00946498, -0.00098538, -0.00525962, -0.00626772]],
                          ms.float32)
    nn_bias = ms.Tensor([0., 0.], ms.float32)
    expect_output = np.array([[0.20668711, -0.531371], [0.4133742, -1.062742], [0., 0.],
                              [0.5373865, -1.3815646], [0.5993926, -1.5409758], [0.5993926, -1.5409758],
                              [0.49604905, -1.2752904]])
    nn_node_feat = ms.Tensor([[1, 2, 3, 4, 1, 2, 3, 4], [2, 4, 1, 3, 2, 4, 1, 3],
                              [1, 3, 2, 4, 1, 3, 2, 4], [9, 7, 5, 8, 9, 7, 5, 8],
                              [8, 7, 6, 5, 8, 7, 6, 5], [8, 6, 4, 6, 8, 6, 4, 6],
                              [1, 2, 1, 1, 1, 2, 1, 1]],
                             ms.float32)
    nn_edge_feat = ones([n_edges, 7], ms.float32)
    edge_func = ms.nn.Dense(edge_feat_size, 2)
    nnconv = NNConv(in_feat_size=8, out_feat_size=2, edge_embed=edge_func)
    nnconv.edge_embed.weight.set_data(nn_weight)
    nnconv.edge_embed.bias.set_data(nn_bias)
    output = nnconv(nn_node_feat, nn_edge_feat, *graph_field.get_graph())
    assert np.allclose(output.asnumpy(), expect_output)


@pytest.mark.lebel0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_stconv():
    """
    Features: STGCN
    Description: Test STGCN.
    Expectation: The output is as expected.
    """
    temp_n_nodes = 4
    temp_n_edges = 6
    feat_size = 2
    edge_attr = ms.Tensor([1, 1, 1, 1, 1, 1], ms.float32)
    edge_index = ms.Tensor([[1, 1, 2, 2, 3, 3],
                            [0, 2, 1, 3, 0, 1]], ms.int32)
    edge_index, edge_weight = norm(edge_index, temp_n_nodes, edge_attr, 'sym')
    edge_weight = ms.ops.Reshape()(edge_weight, ms.ops.Shape()(edge_weight) + (1,))
    batch_size = 2
    input_time_steps = 5
    feat = ms.Tensor(np.ones((batch_size, input_time_steps, temp_n_nodes, feat_size)), ms.float32)
    temp_graph_field = GraphField(edge_index[0], edge_index[1], temp_n_nodes, temp_n_edges)
    stconv = STConv(num_nodes=temp_n_nodes, in_channels=feat_size,
                    hidden_channels=3, out_channels=2,
                    kernel_size=2, k=2)
    temporala_conv1_weight = ms.Tensor([[[[0.0279352, -0.00737858]], [[-0.0593179, -0.119695]]],
                                        [[[0.0072942, 0.01535981]], [[0.0017963, 0.1480704]]],
                                        [[[0.1366018, 0.00844018]], [[-0.0583535, -0.0318455]]]],
                                       ms.float32)
    temporala_conv1_bias = ms.Tensor([0., 0., 0.], ms.float32)

    temporala_conv2_weight = ms.Tensor([[[[0.1577014, -0.1547601]], [[-0.0225174, 0.1132734]],
                                         [[-0.0274447, -0.0769122]]],
                                        [[[0.1160615, 0.1480638]], [[0.2536808, -0.0564105]],
                                         [[-0.0875263, -0.0094614]]]],
                                       ms.float32)
    temporala_conv2_bias = ms.Tensor([0., 0.], ms.float32)
    stconv.temporala_conv1.conv_1.weight.set_data(temporala_conv1_weight)
    stconv.temporala_conv1.conv_1.bias.set_data(temporala_conv1_bias)
    stconv.temporala_conv1.conv_2.weight.set_data(temporala_conv1_weight)
    stconv.temporala_conv1.conv_2.bias.set_data(temporala_conv1_bias)
    stconv.temporala_conv1.conv_3.weight.set_data(temporala_conv1_weight)
    stconv.temporala_conv1.conv_3.bias.set_data(temporala_conv1_bias)
    stconv.temporala_conv2.conv_1.weight.set_data(temporala_conv2_weight)
    stconv.temporala_conv2.conv_1.bias.set_data(temporala_conv2_bias)
    stconv.temporala_conv2.conv_2.weight.set_data(temporala_conv2_weight)
    stconv.temporala_conv2.conv_2.bias.set_data(temporala_conv2_bias)
    stconv.temporala_conv2.conv_3.weight.set_data(temporala_conv2_weight)
    stconv.temporala_conv2.conv_3.bias.set_data(temporala_conv2_bias)
    lin_weight = ms.Tensor([[0.0933186, -0.0433377, -0.10607],
                            [-0.015069, 0.0280074, -0.0451603],
                            [-0.093138, -0.0918521, 0.1034308]],
                           ms.float32)
    lin_bias = ms.Tensor([0., 0., 0.], ms.float32)
    stconv.cheb_conv.lins[0].weight.set_data(lin_weight)
    stconv.cheb_conv.lins[0].bias.set_data(lin_bias)
    stconv.cheb_conv.lins[1].weight.set_data(lin_weight)
    stconv.cheb_conv.lins[1].bias.set_data(lin_bias)
    output = stconv(feat, edge_weight, *temp_graph_field.get_graph())
    expected = np.array([[[[0.02330429, 0.06718753],
                           [0.02330429, 0.06718753],
                           [0.02330429, 0.06718753],
                           [0.02330429, 0.06718753]],
                          [[0.02330429, 0.06718753],
                           [0.02330429, 0.06718753],
                           [0.02330429, 0.06718753],
                           [0.02330429, 0.06718753]],
                          [[0.02330429, 0.06718753],
                           [0.02330429, 0.06718753],
                           [0.02330429, 0.06718753],
                           [0.02330429, 0.06718753]]],
                         [[[0.02330429, 0.06718753],
                           [0.02330429, 0.06718753],
                           [0.02330429, 0.06718753],
                           [0.02330429, 0.06718753]],
                          [[0.02330429, 0.06718753],
                           [0.02330429, 0.06718753],
                           [0.02330429, 0.06718753],
                           [0.02330429, 0.06718753]],
                          [[0.02330429, 0.06718753],
                           [0.02330429, 0.06718753],
                           [0.02330429, 0.06718753],
                           [0.02330429, 0.06718753]]]],
                        )
    assert np.allclose(output.asnumpy(), expected)


@pytest.mark.lebel0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gmmconv():
    """
    Features: GMMGCN
    Description: Test GMMGCN.
    Expectation: The output is as expected.
    """
    gmmconv_weight = ms.Tensor([[1.1284721, 0.672081, 0.30856842, 0.91581726],
                                [-1.0442057, 0.44446903, 0.19224532, 0.36785045],
                                [1.1180657, 0.22537223, 0.48853874, -1.2038152],
                                [-0.8005802, 0.23091288, -0.28297192, 0.46735463]],
                               ms.float32)
    gmmconv_mu = ms.Tensor([[0.01082377, -0.00251063, -0.0095513],
                            [0.05533088, 0.06268983, -0.05767883]],
                           ms.float32)
    expect_output = np.array([[1.1725872, 0.7599088], [2.3262248, 1.3424462], [0., 0.],
                              [6.245813, -1.4796494], [6.6202974, -1.3393956], [6.3376226, -1.3007183],
                              [5.5952363, -1.3932045]]
                             )
    gmmconv = GMMConv(in_feat_size=4, out_feat_size=2, coord_dim=3, n_kernels=2)
    gmmconv.dense.weight.set_data(gmmconv_weight)
    gmmconv.mu.set_data(gmmconv_mu)
    pseudo = ones((8, 3), ms.float32)
    output = gmmconv(node_feat, pseudo, *graph_field.get_graph())
    assert np.allclose(output.asnumpy(), expect_output)


def test_meanconv():
    """
    Features:   MEANConv
    Description: Test MEANConv.
    Expectation: The output is as expected.
    """
    mean_weight = ms.Tensor([[0.05013876, -0.401284615, -0.20843552, -0.00340426, 0.44498375,
                              -0.11601157, 0.6409544, 0.36923417],
                             [0.018585166, -0.06988513, 0.006354237, 0.3507918, -0.12042236,
                              -0.2966982, -0.08903334, 0.9126983]],
                            ms.float32)
    expect_output = np.array([[1.4644403, 3.7632544], [1.5107683, 3.3824015], [0., 1.2248055],
                              [5.0125313, 3.5052013], [0.7575699, 2.6947372], [6.490653, 5.5552692],
                              [6.6787524, 2.6124494]])
    self_idx = ms.Tensor([0, 1, 2, 3, 4, 5, 6], ms.int32)
    meanconv = MeanConv(in_feat_size=4, out_feat_size=2, activation='relu', feat_drop=1.)
    meanconv.dense_self.weight.set_data(mean_weight)
    output = meanconv(node_feat, self_idx, *graph_field.get_graph())
    assert np.allclose(output.asnumpy(), expect_output)


@pytest.mark.lebel0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ginconv():
    """
    Features:   GINConv
    Description: Test GINConv.
    Expectation: The output is as expected.
    """
    expect_output = np.array([[2., 5., 5., 8.], [4., 9., 6., 11.], [1., 3., 2., 4.],
                              [17., 14., 11., 13.], [17., 15., 11., 12.], [17., 13., 9., 14.],
                              [9., 8., 5., 7.]])
    edges_weight = ones((n_edges, 4), ms.float32)
    conv = GINConv(activation=None, init_eps=0., learn_eps=False, aggregation_type="sum")
    output = conv(node_feat, edges_weight, *graph_field.get_graph())
    assert np.allclose(output.asnumpy(), expect_output)


@pytest.mark.lebel0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gcnconv():
    """
    Features:   GCNConv
    Description: Test GCNConv.
    Expectation: The output is as expected.
    """
    expect_output = np.array([[0.78969425, -0.7038187], [1.1158828, -0.8802371], [0., 0.],
                              [-2.8770034, 5.501036], [-1.644399, 2.4849796], [-3.8468409, 4.322275],
                              [-2.596011, 3.065707]])
    gcn_weight = ms.Tensor([[-0.9653046, 0.502546, 0.17415217, 0.05653964],
                            [0.8662434, -0.10252704, 0.35179812, -0.5644021]],
                           ms.float32)
    gcnconv = GCNConv(in_feat_size=4, out_size=2, activation=None, dropout=1.0)
    gcnconv.fc.weight.set_data(gcn_weight)
    output = gcnconv(node_feat, in_degree, out_degree, *graph_field.get_graph())
    assert np.allclose(output.asnumpy(), expect_output)

def test_sgconv():
    """
    Features:    SGConv
    Description: Test GCNConv.
    Expectation: The output is as expected.
    """
    context.set_context(device_target="GPU", mode=context.PYNATIVE_MODE)
    expect_output = np.array([[0.78969425, -0.7038187], [1.1158828, -0.8802371],
                              [0., 0.], [-2.8770034, 5.501037],
                              [-1.6443993, 2.48498], [-3.8468409, 4.322275],
                              [-2.596011, 3.065707]])
    sg_weight = ms.Tensor([[-0.9653046, 0.502546, 0.17415217, 0.05653964],
                           [0.8662434, -0.10252704, 0.35179812, -0.5644021]],
                          ms.float32)
    sgconv = SGConv(in_feat_size=4, out_feat_size=2)
    sgconv.dense.weight.set_data(sg_weight)
    output = sgconv(node_feat, in_degree, out_degree, *graph_field.get_graph())
    assert np.allclose(output.asnumpy(), expect_output)
