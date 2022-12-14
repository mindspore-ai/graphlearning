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
""" test egconv """
import pytest
import numpy as np
import mindspore as ms
from mindspore_gl import GraphField
from mindspore_gl.nn import EGConv

feat_np = np.array([
    [0.6964692, 0.28613934, 0.22685145, 0.5513148],
    [0.71946895, 0.42310646, 0.9807642, 0.6848297],
    [0.4809319, 0.39211753, 0.343178, 0.7290497],
    [0.43857226, 0.0596779, 0.39804426, 0.7379954]
])

expected_symnorm = np.array([
    [0.1819973, -0.30788293, -0.09732638, -0.1594701, -0.00558157, -0.15145838],
    [0.14062165, -0.39629287, -0.0599983, -0.12123133, 0.23993844, -0.08885616],
    [0.29221886, -0.18220055, -0.04745148, -0.15159181, 0.07675514, -0.11340623],
    [0.03700761, -0.1295716, -0.1255244, -0.07247435, 0.08170294, -0.12918073]
])

expected_sum = np.array([
    [0.5621238, -0.86799246, -0.24993664, -0.4494648, -0.04625358, -0.41057765],
    [0.4300964, -1.0662723, -0.13291678, -0.33996275, 0.6487514, -0.22202203],
    [0.68047863, -0.4237072, -0.10731898, -0.35616505, 0.17736161, -0.24043106],
    [0.0740152, -0.2591432, -0.25104883, -0.1449487, 0.1634059, -0.25836146]
])

expected_mean = np.array([
    [0.18737459, -0.2893308, -0.08331221, -0.1498216, -0.01541786, -0.13685922],
    [0.14336546, -0.35542408, -0.0443056, -0.11332091, 0.21625045, -0.07400735],
    [0.34023932, -0.2118536, -0.05365949, -0.17808253, 0.0886808, -0.12021553],
    [0.0370076, -0.1295716, -0.12552442, -0.07247435, 0.08170295, -0.12918073]
])

expected_var = np.array([
    [0.06453722, -0.00278466, -0.0106687, -0.01042172, 0.01114141, 0.00286605],
    [0.0671753, -0.01053588, 0.03240666, -0.00984712, 0.0133568, 0.00789826],
    [0.03805029, -0.00517267, 0.00268535, -0.01749274, -0.00107858, 0.00686282],
    [0.00916045, -0.0013146, -0.00104869, -0.00473798, -0.00057197, 0.00176207]
])

expected_std = np.array([
    [0.24600473, 0.02963194, -0.06901008, -0.06333716, 0.08297312, 0.0342263],
    [0.23170273, -0.01568768, 0.07802369, -0.04363902, 0.09817694, 0.06457907],
    [0.18317135, 0.01929237, -0.01401829, -0.07146345, 0.04134956, 0.02984976],
    [0.0872096, 0.01627139, -0.01870166, -0.04090713, 0.01082803, 0.02289465]
])

expected_max = np.array([
    [4.97252822e-01, -2.53348142e-01, -1.30312771e-01, -2.28138119e-01, 3.26334983e-02, -8.97432864e-02],
    [4.66722548e-01, -3.72857898e-01, 7.65570402e-02, -1.62372738e-01, 3.32553178e-01, 3.08722258e-04],
    [5.23373842e-01, -1.92828104e-01, -6.76611066e-02, -2.49475151e-01, 1.29977643e-01, -9.08112228e-02],
    [1.24160103e-01, -1.13446504e-01, -1.44187614e-01, -1.13359027e-01, 9.24620330e-02, -1.06359832e-01]
])

expected_min = np.array([
    [-0.098125, -0.32442397, 0.02334119, -0.07353501, -0.14716056, -0.17161591],
    [-0.09237397, -0.33332676, -0.11100034, -0.05615281, 0.09497178, -0.15610282],
    [0.15710478, -0.23087908, -0.03965787, -0.10668992, 0.04738396, -0.14961985],
    [-0.0501449, -0.14569671, -0.10686121, -0.03158969, 0.07094386, -0.15200163]
])

expected_multi = np.array([
    [1.0826888, -4.148605, -1.2291584, 0.5051975, 0.571015, 1.0381652],
    [-0.302835, -3.681884, -1.8527321, 0.5714109, 0.2873616, 0.35766196],
    [0.7980148, -2.9009504, -1.1120372, 0.33588743, 0.48472714, 1.0872033],
    [0.33583266, -1.8444395, -0.85494244, 0.69906896, 0.3069592, 0.37023064]
])

expected = {
    'symnorm': expected_symnorm,
    'sum': expected_sum,
    'mean': expected_mean,
    'var': expected_var,
    'std': expected_std,
    'max': expected_max,
    'min': expected_min,
    'multi': expected_multi
}

basis_fc_weight_single = np.array(
    [[0.36272746, 0.679897, 0.4734632, -0.548517],
     [-0.62440497, 0.32165307, 0.01739632, 0.31753114],
     [-0.7568982, -0.04618378, 0.54627943, 0.35934424],
     [0.02828624, 0.15230045, -0.07333887, -0.42589924],
     [-0.29271954, -0.47171366, 0.64338714, 0.42625022],
     [0.270894, -0.5940014, 0.5976957, 0.24286915]]
)
combine_fc_weight_single = np.array(
    [[0.34589732, -0.19669503, 0.10599846, 0.4881912],
     [0.33632958, 0.40099835, -0.10499239, 0.3809386],
     [-0.3916161, 0.04324448, -0.28152746, -0.11660761],
     [-0.12804872, 0.03740954, 0.45506984, 0.24754196],
     [-0.00211811, 0.35492277, -0.2562278, 0.25768572],
     [-0.04636633, -0.08700544, 0.05853671, -0.38304836],
     [0.05777866, 0.16811901, 0.42746586, -0.15567154],
     [0.18001378, 0.49975604, -0.21448451, 0.4752962],
     [-0.24824601, 0.22039747, 0.19587505, 0.13966405]]
)
combine_fc_bias_single = np.array(
    [0.39540154, -0.20208222, 0.13140953,
     0.00280607, -0.3760708, -0.12140697,
     -0.33391154, 0.22107768, 0.04494798]
)

basis_fc_weight_multi = np.array([
    [0.44435292, 0.16143362, 0.7492013, -0.5509026],
    [0.6211722, 0.653935, 0.62627506, 0.11053094],
    [0.7043248, 0.51730275, 0.5779106, -0.05031845],
    [-0.59442407, -0.0096391, 0.14526652, -0.5276313],
    [-0.44433898, -0.7426418, -0.2715329, 0.6747268],
    [0.132453, -0.04731244, 0.03110628, 0.4829964]
])

combine_fc_weight_multi = np.array([
    [-0.44153178, -0.38579482, -0.16623521, -0.28775907],
    [0.25789255, 0.35329175, -0.4851129, -0.42433983],
    [-0.4869089, 0.18862724, 0.40242726, -0.38767004],
    [-0.23147821, 0.15911031, -0.32649267, 0.42473978],
    [0.11658061, -0.1391918, 0.03247529, 0.15588427],
    [-0.17680657, -0.38742793, 0.00336081, 0.00910747],
    [0.0100826, -0.07296515, 0.32103676, -0.13952959],
    [-0.04839635, 0.20559382, -0.31471866, 0.1338793],
    [-0.11055785, 0.23983675, -0.2712322, 0.01847917],
    [0.04891467, -0.40229827, -0.3635711, 0.19175667],
    [-0.14552826, 0.2969483, -0.4939313, -0.24716431],
    [-0.41183096, 0.1997357, -0.01449567, -0.09326071],
    [-0.08318567, -0.39082444, 0.14179748, 0.01246291],
    [-0.34505647, 0.18814385, -0.01004952, -0.4835738],
    [0.26896012, 0.26744354, -0.09420508, -0.34521908],
    [0.02007502, 0.3772899, 0.4576699, -0.37740052],
    [-0.22583449, 0.3893122, 0.24438912, 0.30950648],
    [-0.24886715, 0.43076724, -0.41103297, -0.02408445],
    [0.01038146, 0.0839839, -0.37729198, 0.4587332],
    [0.4913866, -0.3453037, 0.01847041, -0.26633108],
    [0.4794423, 0.27883285, 0.29445034, 0.16130227],
    [-0.0497784, 0.2815312, 0.00851411, -0.18235761],
    [0.2581541, 0.15690315, -0.12955731, -0.13702804],
    [-0.4421698, -0.13705921, -0.20255709, -0.27252442],
    [-0.4516092, 0.39163804, -0.44684577, 0.49635726],
    [-0.2622614, -0.03843355, 0.407871, 0.16495568],
    [-0.14274967, -0.40252846, -0.20442766, 0.40265304]
])

combine_fc_bias_multi = np.array(
    [-0.18880486, 0.41671044, -0.08610827, -0.06376678, 0.19955611, -0.07346714,
     -0.00421083, 0.34629655, 0.16709918, -0.01990223, 0.19041461, 0.43547195,
     0.12602013, -0.14662606, 0.16383564, -0.04374474, -0.390912, -0.19306427,
     0.2274068, 0.01639706, 0.18447214, -0.2926525, 0.4726678, -0.20866126,
     0.10662901, -0.24431562, -0.24116582]
)

conv_bias = np.array([0., 0., 0., 0., 0., 0.])


def do_egconv(aggregators, basis_fc_weight, combine_fc_weight, combine_fc_bias, bias):
    """
    Features: EGConv
    Description: Test EGConv with single aggregator
    Expectation: The output is as expected.
    """
    n_nodes = 4
    n_edges = 7
    num_heads = 3
    num_bases = 3
    input_channels = 4
    output_channels = 6

    src_idx = ms.Tensor([0, 1, 1, 2, 2, 3, 3, 1, 2, 3], ms.int32)
    dst_idx = ms.Tensor([0, 0, 2, 1, 3, 0, 1, 1, 2, 3], ms.int32)
    x = ms.Tensor(feat_np, ms.float32)
    graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)

    conv = EGConv(input_channels, output_channels, aggregators, num_heads, num_bases)
    conv.basis_fc.weight.set_data(ms.Tensor(basis_fc_weight, dtype=ms.float32))
    conv.combine_fc.weight.set_data(ms.Tensor(combine_fc_weight, dtype=ms.float32))
    conv.combine_fc.bias.set_data(ms.Tensor(combine_fc_bias, dtype=ms.float32))
    conv.bias.set_data(ms.Tensor(bias, dtype=ms.float32))

    res = conv(x, *graph_field.get_graph())
    if len(aggregators) == 1:
        assert np.allclose(res.asnumpy(), expected[aggregators[0]])
    else:
        assert np.allclose(res.asnumpy(), expected['multi'])


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("aggr", [['symnorm'], ['sum'], ['mean'], ['var'], ['std'], ['max'], ['min']])
def test_egconv_single_aggr(aggr):
    """
    Features: EGConv
    Description: Test EGConv with single aggregator
    Expectation: The output is as expected.
    """
    do_egconv(aggr,
              basis_fc_weight_single,
              combine_fc_weight_single,
              combine_fc_bias_single,
              conv_bias)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_egconv_multi_aggr():
    """
    Features: EGConv
    Description: Test EGConv with multi aggregator
    Expectation: The output is as expected.
    """
    aggr = ['sum', 'mean', 'max']
    do_egconv(aggr,
              basis_fc_weight_multi,
              combine_fc_weight_multi,
              combine_fc_bias_multi,
              conv_bias)
