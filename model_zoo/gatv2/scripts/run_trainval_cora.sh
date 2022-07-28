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

# data file path: /your/path/cora_v2_with_mask.npz  data_path = /your/path/

python trainval_cora.py --epochs=200 --num_layers=1 --num_hidden=8 --num_heads=8 --num_out_heads=1 \
                        --in_drop=0.6 --attn_drop=0.6 --negative_slope=0.6 \
                        --lr=0.005 --weight_decay=5e-4 --data_path="/your/path/"