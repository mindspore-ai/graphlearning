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
""" test gat """
import os
import shutil
import pytest


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gat():
    """
    Features: GAT
    Description: Test GAT with cora.
    Expectation: The accuracy after training is greater than 0.78.
    """
    if not os.path.exists('./ci_temp'):
        os.mkdir('ci_temp')
    if not os.path.exists('./ci_temp/coo'):
        os.mkdir('ci_temp/coo')
    if os.path.exists('ci_temp/coo/gat'):
        shutil.rmtree('ci_temp/coo/gat')

    cmd_copy = "cp -r ../../model_zoo/gat/ ./ci_temp/coo"
    os.system(cmd_copy)

    cmd_train = "python ./ci_temp/coo/gat/trainval_cora.py " \
                "--data_path=\"/home/workspace/mindspore_dataset/GNN_Dataset/\" >> ./ci_temp/coo/gat/trainval_cora.log"
    os.system(cmd_train)

    file = open("./ci_temp/coo/gat/trainval_cora.log", "r")
    log_info = file.readlines()
    file.close()
    last_info = log_info[-1]
    test_acc = float(last_info[last_info.find('test_acc:'):].replace('test_acc:', '').replace('\n', ''))
    assert test_acc > 0.78

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gat_csr():
    """
    Features: GAT csr
    Description: Test csr GAT with cora.
    Expectation: The accuracy after training is greater than 0.76.
    """
    if not os.path.exists('./ci_temp'):
        os.mkdir('ci_temp')
    if not os.path.exists('./ci_temp/csr'):
        os.mkdir('ci_temp/csr')
    if not os.path.exists('ci_temp/csr/gat'):
        cmd_copy = "cp -r ../../model_zoo/gat/ ./ci_temp/csr/"
        os.system(cmd_copy)

    cmd_train = "python ./ci_temp/csr/gat/trainval_cora.py  --fuse True --csr True " \
                "--data_path=\"/home/workspace/mindspore_dataset/GNN_Dataset/\" >>" \
                " ./ci_temp/csr/gat/trainval_cora_csr.log"
    os.system(cmd_train)

    file = open("./ci_temp/csr/gat/trainval_cora_csr.log", "r")
    log_info = file.readlines()
    file.close()
    last_info = log_info[-1]
    test_acc = float(last_info[last_info.find('test_acc:'):].replace('test_acc:', '').replace('\n', ''))
    assert test_acc > 0.76
    