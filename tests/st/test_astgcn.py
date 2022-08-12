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
""" test ASTGCN """
import random
import pytest
import numpy as np
import networkx as nx

import mindspore as ms
from mindspore import Tensor
from mindspore.common import set_seed

from mindspore_gl import GraphField
from mindspore_gl.nn.temporal import ASTGCN
from mindspore_gl.graph import norm

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_astgcn_spatial_attention():
    """
    Features: SpatialAttention
    Description: Test SpatialAttention in ASTGCN
    Expectation: The output is as expected.
    """

    fixed_seed = 123
    np.random.seed(fixed_seed)
    random.seed(fixed_seed)
    set_seed(fixed_seed)

    node_count = 4  # num_of_vertices
    num_for_predict = 5
    len_input = 4  # num_of_timesteps
    nb_time_strides = 1
    node_features = 2  # in_channels
    nb_block = 1
    k = 3
    nb_chev_filter = 8
    nb_time_filter = 8
    batch_size = 2
    normalization = "sym"
    bias = True

    x = np.random.rand(batch_size, node_count, node_features, len_input)
    model = ASTGCN(
        nb_block,
        node_features,
        k,
        nb_chev_filter,
        nb_time_filter,
        nb_time_strides,
        num_for_predict,
        len_input,
        node_count,
        normalization,
        bias,
    )
    spa_att = model.blocks[0].spatial_attention
    output = spa_att(Tensor(x, dtype=ms.float32))

    expected = np.array(
        [
            [
                [0.20377852, 0.22166683, 0.2449799, 0.21835418],
                [0.2596376, 0.2703423, 0.2648422, 0.23241569],
                [0.40095496, 0.35735828, 0.34409145, 0.40320826],
                [0.13562888, 0.1506327, 0.14608642, 0.14602192]],
            [
                [0.20378195, 0.2216757, 0.24498497, 0.21834166],
                [0.2596419, 0.2703557, 0.2648495, 0.23240069],
                [0.40095493, 0.3573218, 0.3440674, 0.40326348],
                [0.13562125, 0.1506468, 0.14609809, 0.14599413]
            ]
        ]
    )
    assert np.allclose(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_astgcn_temporal_attention():
    """
    Features: TemporalAttention
    Description: Test TemporalAttention in ASTGCN
    Expectation: The output is as expected.
    """

    fixed_seed = 123
    np.random.seed(fixed_seed)
    random.seed(fixed_seed)
    set_seed(fixed_seed)

    node_count = 4  # num_of_vertices
    num_for_predict = 5
    len_input = 4  # num_of_timesteps
    nb_time_strides = 1
    node_features = 2  # in_channels
    nb_block = 1
    k = 3
    nb_chev_filter = 8
    nb_time_filter = 8
    batch_size = 2
    normalization = "sym"
    bias = True

    x = np.random.rand(batch_size, node_count, node_features, len_input)
    model = ASTGCN(
        nb_block,
        node_features,
        k,
        nb_chev_filter,
        nb_time_filter,
        nb_time_strides,
        num_for_predict,
        len_input,
        node_count,
        normalization,
        bias,
    )
    temp_att = model.blocks[0].temporal_attention
    output = temp_att(Tensor(x, dtype=ms.float32))

    # The result of fixing random seed to 123
    expected = np.array([
        [[0.22463863, 0.23349608, 0.23915096, 0.23045641],
         [0.22208546, 0.21321405, 0.20550436, 0.20168155],
         [0.14274438, 0.13829152, 0.1267308, 0.12128693],
         [0.41053152, 0.41499838, 0.42861393, 0.4465751]],
        [[0.22466096, 0.23355903, 0.23919019, 0.23043677],
         [0.22208345, 0.2132401, 0.20549819, 0.20167412],
         [0.14274548, 0.13833342, 0.126727, 0.12127904],
         [0.4105101, 0.4148675, 0.42858458, 0.44661006]]
    ])
    assert np.allclose(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_astgcn_chebconv_attention():
    """
    Features: ChebConvAttention
    Description: Test ChebConvAttention in ASTGCN
    Expectation: The output is as expected.
    """

    fixed_seed = 123
    np.random.seed(fixed_seed)
    random.seed(fixed_seed)
    set_seed(fixed_seed)

    node_count = 4  # num_of_vertices
    num_for_predict = 5
    len_input = 4  # num_of_timesteps
    nb_time_strides = 1
    node_features = 2  # in_channels
    nb_block = 1
    k = 3
    nb_chev_filter = 2
    nb_time_filter = 2
    batch_size = 2
    normalization = "sym"
    bias = True

    x = np.random.rand(batch_size, node_count, node_features)
    edge_index = np.array([
        [1, 1, 2, 2, 3, 3],
        [0, 2, 1, 3, 0, 1]
    ])
    x_tilde = np.random.rand(batch_size, node_count, node_count)

    edge_index_norm, edge_weight_norm = norm(Tensor(edge_index, dtype=ms.int32), node_count)
    gf = GraphField(edge_index_norm[1], edge_index_norm[0], node_count, len(edge_index_norm[0]))

    model = ASTGCN(
        nb_block,
        node_features,
        k,
        nb_chev_filter,
        nb_time_filter,
        nb_time_strides,
        num_for_predict,
        len_input,
        node_count,
        normalization,
        bias,
    )
    cheb_attention = model.blocks[0].chebconv_attention

    out = ms.ops.Zeros()((batch_size, node_count, nb_chev_filter), ms.float32)
    for b in range(batch_size):
        out[b] = cheb_attention(
            Tensor(x[b], dtype=ms.float32),
            Tensor(edge_weight_norm, dtype=ms.float32),
            Tensor(x_tilde[b], dtype=ms.float32),
            *gf.get_graph()
        )

    # The result of fixing random seed to 123
    expected = np.array([
        [
            [-0.08943871, -0.15104005],
            [0.29350495, -0.25151926],
            [-0.18821171, -0.080823],
            [-0.13449574, -0.3289123]
        ],
        [
            [-0.02808032, -0.2566945],
            [0.3818179, -0.38402647],
            [-0.21748269, 0.03677787],
            [0.44292647, -0.5152193]
        ]
    ])
    assert np.allclose(out.asnumpy(), expected)


def create_mock_data(number_of_nodes, edge_per_node, in_channels):
    """
    Creating a mock feature matrix and edge index.
    """
    graph = nx.watts_strogatz_graph(number_of_nodes, edge_per_node, 0.5)
    edge_index = np.array([edge for edge in graph.edges()]).T
    x = np.random.uniform(-1, 1, (number_of_nodes, in_channels))
    return x, edge_index


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_astgcn():
    """
    Features: ASTGCN
    Description: Test ASTGCN
    Expectation: The output is as expected.
    """

    fixed_seed = 123
    set_seed(fixed_seed)
    np.random.seed(fixed_seed)
    random.seed(fixed_seed)

    node_count = 9
    edge_per_node = 5
    num_for_predict = 5
    len_input = 5
    nb_time_strides = 1
    node_features = 2
    nb_block = 2
    k = 3
    nb_chev_filter = 8
    nb_time_filter = 8
    batch_size = 2
    normalization = "sym"
    bias = True
    time = len_input

    x_seq = np.zeros([batch_size, node_count, node_features, time])
    edge_index_seq = []
    for b in range(batch_size):
        for t in range(time):
            x, edge_index = create_mock_data(node_count, edge_per_node, node_features)
            x_seq[b, :, :, t] = x
            if b == 0:
                edge_index_seq.append(Tensor(edge_index, dtype=ms.float32))

    model = ASTGCN(
        nb_block,
        node_features,
        k,
        nb_chev_filter,
        nb_time_filter,
        nb_time_strides,
        num_for_predict,
        len_input,
        node_count,
        normalization,
        bias,
    )

    edge_index_norm, edge_weight_norm = norm(
        Tensor(edge_index_seq[0], dtype=ms.int32),
        node_count
    )
    graph_filed = GraphField(
        edge_index_norm[1],
        edge_index_norm[0],
        node_count,
        len(edge_index_norm[0])
    )

    x_seq = Tensor(x_seq, dtype=ms.float32)
    output = model(
        x_seq,
        edge_weight_norm,
        *graph_filed.get_graph()
    )

    expected = np.array([
        [
            [0.07544235, 0.01770892, -0.00136762, -0.01375367, -0.0291306],
            [-0.00164268, -0.07544971, -0.07453038, -0.05700007, -0.08689568],
            [0.08444241, -0.14039081, -0.06733727, -0.05182562, 0.02702163],
            [0.03523396, 0.01553513, 0.03695836, 0.09257778, -0.08981594],
            [0.05687752, -0.05667856, 0.02979729, 0.04449628, -0.13441558],
            [-0.01049405, 0.00867158, -0.04098717, -0.03241039, -0.05956889],
            [0.0408081, 0.01124942, 0.1034631, 0.02151221, -0.00723902],
            [0.01330323, -0.06195316, -0.0469834, -0.05417178, -0.00927808],
            [0.01516541, -0.05664134, 0.03989137, -0.12862343, -0.08578177]
        ],
        [
            [0.06488001, -0.05333024, 0.02698999, 0.01171654, -0.06728228],
            [0.03014321, 0.00812372, 0.09495642, 0.06645866, -0.09272557],
            [0.1029548, -0.02238976, 0.01579647, 0.09963708, -0.06635459],
            [-0.04123358, 0.09782516, 0.05238448, -0.03682288, -0.02022056],
            [0.03939507, -0.103185, -0.10386664, -0.11043143, 0.06707023],
            [0.09731215, 0.03894726, 0.07656047, 0.04925899, -0.02354186],
            [0.06340364, -0.11006572, 0.01950714, -0.01281237, -0.08171634],
            [0.10757576, -0.05676516, 0.01626567, -0.02762957, -0.03414847],
            [-0.02543246, -0.09827094, -0.09030855, -0.01203547, -0.0335354]
        ]
    ])
    assert np.allclose(output.asnumpy(), expected)
