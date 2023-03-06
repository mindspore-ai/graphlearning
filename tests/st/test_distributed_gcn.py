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
""" test distributed gcn """
import os
import shutil
import pytest


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
def test_distributed_gcn():
    """
    Feature: test distributed gcn on multi-gpu

    Description:test distributed gcn training file
    Directly call the training file and store the training log information

    Expectation:Output log file for model accuracy
    """
    if not os.path.exists('./ci_temp'):
        os.mkdir('ci_temp')
    if not os.path.exists('./ci_temp/dist'):
        os.mkdir('ci_temp/dist')
    if os.path.exists('ci_temp/dist/gcn'):
        shutil.rmtree('ci_temp/dist/gcn')

    cmd_copy = "cp -r ../../model_zoo/gcn/ ./ci_temp/dist/"
    os.system(cmd_copy)
    os.environ['DEVICE_TARGET'] = "GPU"
    cmd_train = "mpirun --allow-run-as-root -n 8 python ./ci_temp/dist/gcn/distributed_trainval_reddit.py" \
                " --data_path /home/workspace/mindspore_dataset/GNN_Dataset/ --epochs 1 >>" \
                "./ci_temp/dist/gcn/trainval_reddit.log"
    ret = os.system(cmd_train)
    assert ret == 0
