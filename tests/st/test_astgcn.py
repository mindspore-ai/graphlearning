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
from mindspore_gl.nn import ASTGCN
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
    expected = np.array([[[0.27625212, 0.28754577, 0.2775065, 0.2637076],
                          [0.10309475, 0.11209958, 0.11792921, 0.10741828],
                          [0.4000435, 0.39550117, 0.395912, 0.40439057],
                          [0.22060959, 0.20485348, 0.20865229, 0.22448353]],
                         [[0.27620086, 0.28745344, 0.27747157, 0.26367548],
                          [0.10311195, 0.11208878, 0.11792479, 0.1074442],
                          [0.4001092, 0.3956428, 0.3959695, 0.4044242],
                          [0.22057803, 0.20481496, 0.20863414, 0.22445616]]])
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
    expected = np.array([[[0.30406776, 0.30062982, 0.30742654, 0.30257964],
                          [0.17203102, 0.18575354, 0.16621527, 0.18180256],
                          [0.12641437, 0.11635165, 0.14456336, 0.12288259],
                          [0.39748684, 0.397265, 0.3817948, 0.3927352]],
                         [[0.30406398, 0.30064258, 0.30741987, 0.30259448],
                          [0.1720317, 0.18570797, 0.16621993, 0.18175867],
                          [0.12641919, 0.11632083, 0.14456907, 0.12284693],
                          [0.39748514, 0.39732867, 0.38179114, 0.39279997]]])
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
    print(out.asnumpy())
    # The result of fixing random seed to 123
    expected = np.array([[[-0.24071819, 0.0393681],
                          [-0.50393456, -0.28954452],
                          [-0.7796291, 0.42713895],
                          [-0.8683217, 0.5765807]],
                         [[-0.32263505, 0.04425192],
                          [-0.4277593, -0.5655104],
                          [-1.1740817, 1.0882314],
                          [-0.7727675, -0.04370432]]])
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
    print(output.asnumpy())
    expected = np.array([
        [
            [0.12332413, -0.60853606, 0.2538968, -0.6431123, -0.8342397],
            [0.37709016, -0.6668014, -0.16025855, -0.47109377, -0.25419474],
            [-0.34617686, -1.0258948, 0.26222453, -0.7868134, -0.35702634],
            [-0.0708486, -0.414764, -0.2594444, -0.23942429, -0.9127646],
            [-0.09841944, -0.5891067, -0.6341287, -0.05767142, -0.81305516],
            [-0.01868853, -0.5334885, 0.42312554, -1.1781822, 0.08392156],
            [-0.15350942, -0.6544952, 0.60715353, -0.6603354, 0.26543146],
            [-0.04985616, -0.79426664, 0.3369154, -0.98899233, -0.5991371],
            [-0.66024303, -0.5499391, 0.43394357, -0.2836772, -0.8950232],
        ],
        [
            [-0.17054589, -0.74592644, 0.10469497, -0.88878226, -0.86839217],
            [-0.39679527, -0.38165516, 0.43686327, -0.11684191, -0.23846848],
            [0.20339325, -0.8422794, -0.6738411, -0.09173622, 0.07141262],
            [-0.3471542, -1.0114025, 0.6390641, -0.5992087, -0.32323796],
            [-0.85847586, -1.0090373, -0.28923938, -0.28811106, -0.31478548],
            [-0.34000725, -0.67790705, 0.13271187, -0.2746954, 0.32104883],
            [-0.7025708, -0.77970237, 0.07166047, -0.71367323, -0.76619583],
            [-0.0484736, -0.01026601, 0.48944473, -1.313022, -0.2449228],
            [0.02068572, -1.2364916, 0.17532463, -0.3978162, -0.41436273],
        ]
    ])
    assert np.allclose(output.asnumpy(), expected, atol=1e-05)
