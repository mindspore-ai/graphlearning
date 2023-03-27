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
""" test gatv2 """
import os
import shutil
import pytest
import numpy as np
import mindspore as ms
from mindspore_gl import GraphField
from mindspore_gl.nn import GATv2Conv

feat_np = np.array([[0.6964692, 0.28613934, 0.22685145, 0.5513148],
                    [0.71946895, 0.42310646, 0.9807642, 0.6848297],
                    [0.4809319, 0.39211753, 0.343178, 0.7290497],
                    [0.43857226, 0.0596779, 0.39804426, 0.7379954]])
fc_s_weight = np.array([[0.12217519, 0.6400273, -0.7319936, -0.52148896],
                        [-0.30849522, 0.0310904, -0.18065873, -0.08512134],
                        [-0.7551467, 0.36272746, 0.679897, 0.4734632],
                        [-0.548517, -0.62440497, 0.32165307, 0.01739632],
                        [0.31753114, -0.7568982, -0.04618378, 0.54627943],
                        [0.35934424, 0.02828624, 0.15230045, -0.07333887]])
fc_s_bias = np.array([-0.27491677, -0.18894964, -0.30448985, 0.4153046, 0.27514333, 0.17486131])
fc_d_weight = np.array([[-0.5940014, 0.5976957, 0.24286915, 0.53586185],
                        [-0.30471864, 0.16421211, 0.75630254, 0.52103955],
                        [0.621224, -0.16265352, 0.59014755, -0.6066891],
                        [0.06699406, -0.43614048, -0.18064773, -0.19837223],
                        [0.05795462, 0.70499116, 0.38349038, -0.00328136],
                        [0.549844, -0.3969464, 0.399205, -0.07183041]])
fc_d_bias = np.array([-0.08700544, 0.05853671, -0.38304836, 0.05777866, 0.16811901, 0.42746586])
attn = np.array([[-0.34105927, 0.39439043],
                 [1.0949106, -0.46991205],
                 [1.0413219, -0.5438798]])

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gatv2conv():
    """
    Features: GATv2Conv
    Description: Test GATv2Conv without self loop.
    Expectation: The output is as expected.
    """
    n_nodes = 4
    n_edges = 7
    num_heads = 3
    out_size = 2

    expected = np.array([[-0.77293885, -0.52461433, 0.01839483, 0.09849139, 0.6280106, 0.42526117],
                         [-0.72894645, -0.45317966, 0.03042316, 0.14938739, 0.6492665, 0.34801626],
                         [-0.9912601, -0.6332251, 0.29673827, 0.08385316, 0.51216155, 0.5445126],
                         [-0.5965884, -0.44917953, 0.05307168, 0.0297322, 0.5134767, 0.35757142]])

    src_idx = ms.Tensor([0, 1, 1, 2, 2, 3, 3], ms.int32)
    dst_idx = ms.Tensor([0, 0, 2, 1, 3, 0, 1], ms.int32)
    feat = ms.Tensor(feat_np, ms.float32)
    graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)

    gatconv = GATv2Conv(in_feat_size=n_nodes, out_size=out_size, num_attn_head=num_heads)
    gatconv.fc_s.weight.set_data(ms.Tensor(fc_s_weight, dtype=ms.float32))
    gatconv.fc_s.bias.set_data(ms.Tensor(fc_s_bias, dtype=ms.float32))
    gatconv.fc_d.weight.set_data(ms.Tensor(fc_d_weight, dtype=ms.float32))
    gatconv.fc_d.bias.set_data(ms.Tensor(fc_d_bias, dtype=ms.float32))
    gatconv.attn.set_data(ms.Tensor(attn, dtype=ms.float32))

    res = gatconv(feat, *graph_field.get_graph())
    assert np.allclose(res.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gatv2conv_self_loop():
    """
    Features: GATv2Conv
    Description: Test GATv2Conv with self loop.
    Expectation: The output is as expected.
    """
    n_nodes = 4
    n_edges = 7
    num_heads = 3
    out_size = 2

    expected = np.array([[-0.77293885, -0.52461433, 0.01839483, 0.09849139, 0.6280106, 0.42526117],
                         [-0.8133176, -0.51108986, 0.13763823, 0.12300415, 0.61059916, 0.40343353],
                         [-0.78996193, -0.53935456, 0.17784499, 0.05744568, 0.51285297, 0.44623053],
                         [-0.7289464, -0.45317966, 0.03047406, 0.14911842, 0.64926654, 0.34801632]])

    src_idx = ms.Tensor([0, 1, 1, 2, 2, 3, 3, 1, 2, 3], ms.int32)
    dst_idx = ms.Tensor([0, 0, 2, 1, 3, 0, 1, 1, 2, 3], ms.int32)
    feat = ms.Tensor(feat_np, ms.float32)
    graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)

    gatconv = GATv2Conv(in_feat_size=n_nodes, out_size=out_size, num_attn_head=num_heads)
    gatconv.fc_s.weight.set_data(ms.Tensor(fc_s_weight, dtype=ms.float32))
    gatconv.fc_s.bias.set_data(ms.Tensor(fc_s_bias, dtype=ms.float32))
    gatconv.fc_d.weight.set_data(ms.Tensor(fc_d_weight, dtype=ms.float32))
    gatconv.fc_d.bias.set_data(ms.Tensor(fc_d_bias, dtype=ms.float32))
    gatconv.attn.set_data(ms.Tensor(attn, dtype=ms.float32))

    res = gatconv(feat, *graph_field.get_graph())
    assert np.allclose(res.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gatv2():
    """
    Features: GATv2
    Description: Test GATv2 with cora.
    Expectation: The accuracy after training is greater than 0.79.
    """
    if not os.path.exists('./ci_temp'):
        os.mkdir('ci_temp')
    if not os.path.exists('./ci_temp/coo'):
        os.mkdir('ci_temp/coo')
    if os.path.exists('ci_temp/coo/gatv2'):
        shutil.rmtree('ci_temp/coo/gatv2')

    cmd_copy = "cp -r ../../model_zoo/gatv2/ ./ci_temp/coo/"
    os.system(cmd_copy)

    cmd_train = "python ./ci_temp/coo/gatv2/trainval_cora.py " \
                "--data_path=\"/home/workspace/mindspore_dataset/GNN_Dataset/\" >>" \
                " ./ci_temp/coo/gatv2/trainval_cora.log"
    os.system(cmd_train)

    file = open("./ci_temp/coo/gatv2/trainval_cora.log", "r")
    log_info = file.readlines()
    file.close()
    last_info = log_info[-1]  # e.g. "epoch: 199  test_acc: 0.822  loss: 1.0377"
    test_acc = float(last_info.split()[3])
    assert test_acc > 0.79

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gatv2_csr():
    """
    Features: GATv2 csr
    Description: Test csr GATv2 with cora.
    Expectation: The accuracy after training is greater than 0.76.
    """
    if not os.path.exists('./ci_temp'):
        os.mkdir('ci_temp')
    if not os.path.exists('./ci_temp/csr'):
        os.mkdir('ci_temp/csr')
    if not os.path.exists('ci_temp/csr/gatv2'):
        cmd_copy = "cp -r ../../model_zoo/gatv2/ ./ci_temp/csr/"
        os.system(cmd_copy)

    cmd_train = "python ./ci_temp/csr/gatv2/trainval_cora.py --fuse True --csr True " \
                "--data_path=\"/home/workspace/mindspore_dataset/GNN_Dataset/\" >>" \
                " ./ci_temp/csr/gatv2/trainval_cora_csr.log"
    os.system(cmd_train)

    file = open("./ci_temp/csr/gatv2/trainval_cora_csr.log", "r")
    log_info = file.readlines()
    file.close()
    last_info = log_info[-1]  # e.g. "epoch: 199  test_acc: 0.822  loss: 1.0377"
    test_acc = float(last_info.split()[3])
    assert test_acc > 0.76
    