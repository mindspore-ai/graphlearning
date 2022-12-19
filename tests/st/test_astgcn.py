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

    expected = np.array(
        [
            [
                [0.28815734, 0.2667385, 0.30206463, 0.31233814],
                [0.2251567, 0.24223456, 0.2267898, 0.2231716],
                [0.21986069, 0.21713911, 0.2102986, 0.20838758],
                [0.26682535, 0.27388784, 0.260847, 0.25610265],
            ],
            [
                [0.28809702, 0.26666203, 0.30201668, 0.31222385],
                [0.22541785, 0.24249779, 0.22700807, 0.22344896],
                [0.21983938, 0.21714815, 0.21026228, 0.20840216],
                [0.26664573, 0.27369207, 0.26071295, 0.2559251],
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
    expected = np.array(
        [
            [
                [0.16080944, 0.16016997, 0.16871159, 0.18589061],
                [0.31302705, 0.34766364, 0.33148938, 0.32554796],
                [0.34384686, 0.39094645, 0.35484582, 0.3480562],
                [0.18231669, 0.10121991, 0.14495319, 0.14050527],
            ],
            [
                [0.16080362, 0.16012932, 0.16868417, 0.18587302],
                [0.3130532, 0.34770727, 0.33151904, 0.32558772],
                [0.3438038, 0.3909557, 0.35483912, 0.3480343],
                [0.18233949, 0.10120774, 0.14495769, 0.14050494],
            ]
        ]
    )
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
    expected = np.array(
        [
            [
                [-0.11579961, -0.0226942],
                [0.15662822, -0.02039342],
                [-0.5163456, -0.65948397],
                [-0.6727781, -0.5804959]
            ],
            [
                [-0.142562, 0.00620593],
                [0.4188207, 0.39029145],
                [-1.1370283, -1.6000638],
                [-0.08905573, -0.10538422]
            ]
        ]
    )
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
            [-0.09041344, -0.06217816, -0.04309423, 0.00356692, 0.01969181],
            [-0.06157371, 0.02655463, 0.09800988, -0.07199652, -0.04025764],
            [0.07671128, -0.019453, -0.08042306, -0.0386402, 0.09097143],
            [0.02155785, 0.04942806, 0.05396117, -0.00949986, 0.00384752],
            [-0.12450423, -0.04623427, 0.01006134, -0.02616193, 0.03322661],
            [0.12445185, -0.01280287, -0.10317003, -0.04593236, -0.03610378],
            [0.02166866, -0.15273258, -0.0426378, 0.09594952, 0.00288021],
            [-0.10615215, -0.02442015, 0.04203781, -0.15040775, -0.05190156],
            [-0.07406512, -0.09904732, -0.03606629, -0.00658559, -0.10576056],
        ],
        [
            [0.06126893, 0.00164253, -0.0247033, -0.05532996, -0.01293815],
            [-0.04142861, -0.01103848, -0.02953731, -0.04202444, 0.06040662],
            [-0.07826433, 0.08688006, -0.01837174, -0.0504397, 0.00369798],
            [-0.00528669, -0.14845428, 0.00532585, -0.03113262, -0.13406053],
            [-0.07397612, 0.06000311, -0.02571755, -0.15438497, 0.08686772],
            [-0.00176271, 0.02551298, -0.01487571, -0.09020602, 0.02317336],
            [0.0914247, -0.03913187, -0.02264144, -0.00770737, -0.00539655],
            [0.01960496, 0.01623925, -0.0474415, -0.04788478, 0.02959474],
            [-0.02865497, 0.05499984, 0.04221002, 0.0196395, -0.05342437]
        ]
    ])
    assert np.allclose(output.asnumpy(), expected)
