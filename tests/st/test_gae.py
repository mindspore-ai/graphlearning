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
""" test gae """
import os
import shutil
import pytest

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def tast_gae():
    """
    Feature:test gae

    Description:test gae training file
    Directly call the training file and store the training log information

    Expectation:Output log file for model accuracy
    """
    if not os.path.exists('./ci_temp'):
        os.mkdir('ci_temp')
    if os.path.exists('ci_temp/gae'):
        shutil.rmtree('ci_temp/gae')

    cmd_copy = "cp -r ../../model_zoo/gae/ ./ci_temp/"
    os.system(cmd_copy)

    cmd_train = "python ./ci_temp/gae/trainval.py --fuse=True " \
                "--data_path=\"/home/workspace/mindspore_dataset/GNN_Dataset/\" >> ./ci_temp/gae/trainval.log"
    os.system(cmd_train)

    file = open("./ci_temp/gae/trainval.log", "r")
    log_info = file.readlines()
    last_info = log_info[-1]
    test_auc = float(last_info[last_info.find('Test Auc score:'):\
                                    last_info.find('AP score:')].replace('Test Auc score:', ''))
    test_ap = float(last_info[last_info.find('AP score:'):].replace('AP score:', ''))
    assert test_auc > 0.7
    assert test_ap > 0.7
