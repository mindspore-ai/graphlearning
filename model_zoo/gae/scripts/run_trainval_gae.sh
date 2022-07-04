#!/bin/bash
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

# data file path: ./data/cora_with_mask.npz  data_path = ./data

python ../trainval.py --epochs=200 --data_name 'cora_v2' --lr=0.01 --weight_decay=0.0 --dropout=0.0 \
                        --hidden1_dim=32 --hidden2_dim 16 --mode "undirected" --data_path "./data"