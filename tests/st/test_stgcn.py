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
""" test stgcn """
import os
import shutil
import pytest


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_stgcn():
    """
    Feature: test stgcn net

    Description:test stgcn training file
    Directly call the training file and store the training log information

    Expectation:Output log file for model accuracy
    """
    if not os.path.exists('./ci_temp'):
        os.mkdir('ci_temp')
    if os.path.exists('ci_temp/stgcn'):
        shutil.rmtree('ci_temp/stgcn')

    cmd_copy = "cp -r ../../model_zoo/stgcn/ ./ci_temp/"
    os.system(cmd_copy)

    cmd_train = "python ./ci_temp/stgcn/trainval_metr.py --epochs 1 " \
                "--data-path \"/home/workspace/mindspore_dataset/GNN_Dataset/METR-LA/\" >>" \
                "./ci_temp/stgcn/trainval_metr.log"
    os.system(cmd_train)

    file = open("./ci_temp/stgcn/trainval_metr.log", "r")
    log_info = file.readlines()
    file.close()
    last_info = log_info[-1]
    test_mse = float(last_info[last_info.find('eval MSE:'):].replace('eval MSE:', '').replace('\n', ''))
    assert test_mse < 0.4
