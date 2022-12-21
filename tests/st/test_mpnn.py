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
""" test mpnn """
import os
import shutil
import pytest


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_mpnn():
    """
    Features: mpnn
    Description: Test mpnn with alchemy
    Expectation: The output is as expected.
    """
    if not os.path.exists('./ci_temp'):
        os.mkdir('ci_temp')
    if os.path.exists('ci_temp/mpnn'):
        shutil.rmtree('ci_temp/mpnn')

    cmd_copy = "cp -r ../../model_zoo/mpnn/ ./ci_temp/"
    os.system(cmd_copy)

    cmd_train = "python ./ci_temp/mpnn/trainval_alchemy.py --data_size 400 --epochs 2 --batch_size 16 " \
                "--data_path=\"/home/workspace/mindspore_dataset/GNN_Dataset/\" >> ./ci_temp/mpnn/trainval_alchemy.log"
    os.system(cmd_train)

    file = open("./ci_temp/mpnn/trainval_alchemy.log", "r")
    log_info = file.readlines()
    file.close()
    if 'Test mae' in log_info[-1]:
        last_info = log_info[-1]
    else:
        last_info = log_info[-2]
    test_mae = float(last_info[last_info.find('Test mae '):].replace('Test mae ', '').replace('\n', ''))
    assert test_mae < 2.0
