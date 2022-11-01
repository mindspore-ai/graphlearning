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
                [-0.11899118, -0.07920256],
                [-0.4080219, -0.48470926],
                [-0.12027679, -0.03094712],
                [-0.37344167, -0.40766814]
            ],
            [
                [-0.22327462, -0.25296563],
                [-0.5073078, -0.67749727],
                [-0.10642301, 0.07816847],
                [-0.74628973, -0.9252911]
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
            [-0.09526052, -0.06221562, -0.03767933, -0.00176416, 0.02042305],
            [-0.05723682, 0.0324668, 0.08839224, -0.06750376, -0.04139298],
            [0.07553598, -0.01705341, -0.09137351, -0.03440098, 0.08545841],
            [0.03336725, 0.05970329, 0.05040623, -0.01599668, -0.00527577],
            [-0.12353964, -0.04716953, 0.01205427, -0.0275861, 0.0307261],
            [0.11641323, -0.02630173, -0.11797275, -0.04230388, -0.02571069],
            [0.00560516, -0.17197853, -0.055987, 0.10054947, 0.02307409],
            [-0.07731892, 0.01840114, 0.0226675, -0.16192885, -0.04055912],
            [-0.07243706, -0.10184192, -0.03762899, -0.00819593, -0.10528144]
        ],
        [
            [0.06413533, 0.0106135, -0.02848716, -0.0554461, -0.0147839],
            [-0.04186962, -0.01346519, -0.03197283, -0.03995561, 0.06077475],
            [-0.08719444, 0.08630574, -0.01570232, -0.06800009, 0.0000526],
            [0.01156713, -0.15793672, -0.00152045, -0.03042351, -0.12498395],
            [-0.0753834, 0.04560315, -0.02792881, -0.14839442, 0.08663175],
            [-0.00470069, 0.02697017, -0.01243714, -0.08664192, 0.02043965],
            [0.09565099, -0.05234185, -0.03651035, 0.00443882, -0.03923392],
            [0.01425175, 0.02266738, -0.03689587, -0.03178114, 0.02946016],
            [-0.02752657, 0.05761655, 0.03785872, 0.02105006, -0.05689116]
        ]
    ])
    assert np.allclose(output.asnumpy(), expected)
