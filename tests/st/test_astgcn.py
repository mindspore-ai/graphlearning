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
    expected = np.array([[[1.15008056e-01, -5.87044179e-01, 2.46746868e-01, -6.61389649e-01, -8.48924041e-01],
                          [3.66957784e-01, -6.55795872e-01, -1.75286084e-01, -4.56361651e-01, -2.66989172e-01],
                          [-3.40422809e-01, -1.02312410e+00, 2.59515196e-01, -7.85708547e-01, -3.54279816e-01],
                          [-7.44204372e-02, -4.08418357e-01, -2.51313388e-01, -2.41524279e-01, -9.23472703e-01],
                          [-1.06734648e-01, -6.15274727e-01, -6.06003284e-01, -4.22899686e-02, -8.05802882e-01],
                          [9.61452723e-04, -5.43396235e-01, 4.06172395e-01, -1.18271637e+00, 1.10309258e-01],
                          [-1.24843612e-01, -6.62699461e-01, 6.42972708e-01, -6.44818664e-01, 1.87746003e-01],
                          [-6.79711401e-02, -7.90324569e-01, 3.34719986e-01, -9.92372751e-01, -5.76270461e-01],
                          [-6.60412610e-01, -5.65945089e-01, 4.49451089e-01, -2.79119968e-01, -8.78782749e-01]],
                         [[-1.70204654e-01, -7.42189288e-01, 1.11341536e-01, -8.95694256e-01, -8.56932998e-01],
                          [-3.61103475e-01, -3.73250842e-01, 4.37343031e-01, -1.23978183e-01, -2.66665518e-01],
                          [2.08283380e-01, -8.45679283e-01, -6.73992515e-01, -8.70067105e-02, 5.46515919e-02],
                          [-3.22467744e-01, -1.00943398e+00, 6.44718766e-01, -6.12795472e-01, -2.98030376e-01],
                          [-8.72556865e-01, -1.00786948e+00, -2.83011109e-01, -2.80067146e-01, -3.27295363e-01],
                          [-2.83719420e-01, -6.51098609e-01, 9.73470211e-02, -2.79922724e-01, 2.61578411e-01],
                          [-7.00510800e-01, -7.96726644e-01, 8.50140452e-02, -7.25900829e-01, -7.39624977e-01],
                          [-4.18395028e-02, -3.19613814e-02, 4.94102329e-01, -1.30454123e+00, -2.48471066e-01],
                          [1.86553523e-02, -1.23263288e+00, 1.52212903e-01, -3.97183120e-01, -3.76585305e-01]]
                         ])
    assert np.allclose(output.asnumpy(), expected)
