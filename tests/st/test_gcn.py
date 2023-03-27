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
""" test gcn """
import os
import shutil
import pytest


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gcn():
    """
    Features: gcn cora csr
    Description: Test gcn with cora
    Expectation: The output is as expected.
    """
    if not os.path.exists('./ci_temp'):
        os.mkdir('ci_temp')
    if not os.path.exists('./ci_temp/coo'):
        os.mkdir('ci_temp/coo')
    if os.path.exists('ci_temp/coo/gcn'):
        shutil.rmtree('ci_temp/coo/gcn')

    cmd_copy = "cp -r ../../model_zoo/gcn/ ./ci_temp/coo/"
    os.system(cmd_copy)

    cmd_train = "python ./ci_temp/coo/gcn/trainval_cora.py " \
                "--data_path=\"/home/workspace/mindspore_dataset/GNN_Dataset/\" >> ./ci_temp/coo/gcn/trainval_cora.log"
    os.system(cmd_train)

    file = open("./ci_temp/coo/gcn/trainval_cora.log", "r")
    log_info = file.readlines()
    file.close()
    last_info = log_info[-1]
    test_acc = float(last_info[last_info.find('best_acc:'):].replace('best_acc:', '').replace('\n', ''))
    assert test_acc > 0.78

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gcn_csr():
    """
    Features: gcn cora csr
    Description: Test csr gcn with cora
    Expectation: The output is as expected.
    """
    if not os.path.exists('./ci_temp'):
        os.mkdir('ci_temp')
    if not os.path.exists('./ci_temp/csr'):
        os.mkdir('ci_temp/csr')
    if os.path.exists('ci_temp/csr/gcn'):
        shutil.rmtree('ci_temp/csr/gcn')

    cmd_copy = "cp -r ../../model_zoo/gcn/ ./ci_temp/csr/"
    os.system(cmd_copy)

    cmd_train = "python ./ci_temp/csr/gcn/trainval_cora.py --fuse True --csr True " \
                "--data_path=\"/home/workspace/mindspore_dataset/GNN_Dataset/\" >> " \
                " ./ci_temp/csr/gcn/trainval_cora_csr.log"
    os.system(cmd_train)

    file = open("./ci_temp/csr/gcn/trainval_cora_csr.log", "r")
    log_info = file.readlines()
    file.close()
    last_info = log_info[-1]
    test_acc = float(last_info[last_info.find('best_acc:'):].replace('best_acc:', '').replace('\n', ''))
    assert test_acc > 0.78

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gcn_reddit_csr():
    """
    Features: gcn reddit csr
    Description: Test csr gcn with reddit
    Expectation: The output is as expected.
    """
    if not os.path.exists('./ci_temp'):
        os.mkdir('ci_temp')
    if not os.path.exists('./ci_temp/csr_reddit'):
        os.mkdir('ci_temp/csr_reddit')
    if os.path.exists('ci_temp/csr_reddit/gcn'):
        shutil.rmtree('ci_temp/csr_reddit/gcn')

    cmd_copy = "cp -r ../../model_zoo/gcn/ ./ci_temp/csr_reddit/"
    os.system(cmd_copy)

    cmd_train = "python ./ci_temp/csr_reddit/gcn/trainval_csr_reddit.py --epochs 10 " \
                "--data_path=\"/home/workspace/mindspore_dataset/GNN_Dataset/\" >>" \
                " ./ci_temp/csr_reddit/gcn/trainval_csr_reddit.log"
    os.system(cmd_train)

    file = open("./ci_temp/csr_reddit/gcn/trainval_csr_reddit.log", "r")
    log_info = file.readlines()
    file.close()
    last_info = log_info[-1]
    test_acc = float(last_info[last_info.find('Test acc:'):].replace('Test acc:', '').replace('\n', ''))
    assert test_acc > 0.59
