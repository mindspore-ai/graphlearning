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
""" test deepwalk """
import os
import shutil
import pytest


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_deepwalk():
    """test deepwalk"""
    if not os.path.exists('./ci_temp'):
        os.mkdir('ci_temp')
    if os.path.exists('ci_temp/deepwalk'):
        shutil.rmtree('ci_temp/deepwalk')

    cmd_copy = "cp -r ../../model_zoo/deepwalk/ ./ci_temp/"
    os.system(cmd_copy)

    cmd_train = "python ./ci_temp/deepwalk/pretrain_blog_catalog.py --epoch 10 " \
                "--data_path=\"/home/workspace/mindspore_dataset/GNN_Dataset/\" " \
                "--save_file_path ./ci_temp/deepwalk/"
    os.system(cmd_train)

    cmd_eval = "python ./ci_temp/deepwalk/trainval_blog_catalog.py --epoch 10 " \
               "--save_file_path ./ci_temp/deepwalk/ " \
               "--data_path=\"/home/workspace/mindspore_dataset/GNN_Dataset/\" " \
                ">> ./ci_temp/deepwalk/trainval_blogcatelog.log"
    os.system(cmd_eval)

    file = open("./ci_temp/deepwalk/trainval_blogcatelog.log", "r")
    log_info = file.readlines()
    file.close()
    last_info = log_info[-1]
    micro_f1 = float(last_info[last_info.find('micro f1:'):last_info.find(', ')].replace('micro f1:', '').
                     replace('\n', ''))
    assert micro_f1 > 0.25
