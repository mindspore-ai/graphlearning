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
""" test examples func scripts """
import os
import shutil
import pytest


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_appnp_func():
    """
    Feature: test appnp func

    Description:test appnp func training file
    Directly call the training file and store the training log information

    Expectation:Output log file for model speed
    """
    if not os.path.exists('./ci_temp'):
        os.mkdir('ci_temp')
    if not os.path.exists('./ci_temp/examples_appnp'):
        os.mkdir('./ci_temp/examples_appnp')
    if os.path.exists('./ci_temp/examples_appnp/examples'):
        shutil.rmtree('./ci_temp/examples_appnp/examples')

    cmd_copy = "cp -r ../../examples/ ./ci_temp/examples_appnp/ &&" \
               "cp -r ../../model_zoo/appnp/ ./ci_temp/examples_appnp/examples"
    os.system(cmd_copy)

    cmd_train = "python ci_temp/examples_appnp/examples/vc_appnp_datanet_func.py --fuse True " \
                "--data-path \"/home/workspace/mindspore_dataset/GNN_Dataset/cora_v2_with_mask.npz\" >>" \
                "./ci_temp/examples_appnp/trainval_appnp_func.log"
    os.system(cmd_train)

    file = open("./ci_temp/examples_appnp/trainval_appnp_func.log", "r")
    log_info = file.readlines()
    file.close()
    last_info = log_info[-1]
    training_speed = float(last_info[last_info.find('epoch time:'):].replace('epoch time:', '').replace('\n', ''))
    assert training_speed < 2.1

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gat_func():
    """
    Feature: test gat func

    Description:test gat func training file
    Directly call the training file and store the training log information

    Expectation:Output log file for model speed
    """
    if not os.path.exists('./ci_temp'):
        os.mkdir('ci_temp')
    if not os.path.exists('./ci_temp/examples_gat'):
        os.mkdir('./ci_temp/examples_gat')
    if os.path.exists('./ci_temp/examples_gat/examples'):
        shutil.rmtree('./ci_temp/examples_gat/examples')

    cmd_copy = "cp -r ../../examples/ ./ci_temp/examples_gat/ &&" \
               "cp -r ../../model_zoo/gat/ ./ci_temp/examples_gat/examples"
    os.system(cmd_copy)

    cmd_train = "python ci_temp/examples_gat/examples/vc_gat_datanet_func.py --fuse True " \
                "--data-path \"/home/workspace/mindspore_dataset/GNN_Dataset/cora_v2_with_mask.npz\" >>" \
                "./ci_temp/examples_gat/trainval_gat_func.log"
    os.system(cmd_train)

    file = open("./ci_temp/examples_gat/trainval_gat_func.log", "r")
    log_info = file.readlines()
    file.close()
    last_info = log_info[-1]
    training_speed = float(last_info[last_info.find('epoch time:'):].replace('epoch time:', '').replace('\n', ''))
    assert training_speed < 2

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gcn_func():
    """
    Feature: test gcn func

    Description:test gcn func training file
    Directly call the training file and store the training log information

    Expectation:Output log file for model speed
    """
    if not os.path.exists('./ci_temp'):
        os.mkdir('ci_temp')
    if not os.path.exists('./ci_temp/examples_gcn'):
        os.mkdir('./ci_temp/examples_gcn')
    if os.path.exists('./ci_temp/examples_gcn/examples'):
        shutil.rmtree('./ci_temp/examples_gcn/examples')

    cmd_copy = "cp -r ../../examples/ ./ci_temp/examples_gcn/ &&" \
               "cp -r ../../model_zoo/gcn/ ./ci_temp/examples_gcn/examples"
    os.system(cmd_copy)

    cmd_train = "python ci_temp/examples_gcn/examples/vc_gcn_datanet_func.py --fuse True " \
                "--data-path \"/home/workspace/mindspore_dataset/GNN_Dataset/cora_v2_with_mask.npz\" >>" \
                "./ci_temp/examples_gcn/trainval_gcn_func.log"
    os.system(cmd_train)

    file = open("./ci_temp/examples_gcn/trainval_gcn_func.log", "r")
    log_info = file.readlines()
    file.close()
    last_info = log_info[-1]
    training_speed = float(last_info[last_info.find('epoch time:'):].replace('epoch time:', '').replace('\n', ''))
    assert training_speed < 2
