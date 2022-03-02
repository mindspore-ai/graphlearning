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

for name in CoraFull.npz AmazonCoBuy_computers.npz Coauthor_physics.npz Coauthor_cs.npz pubmed_with_mask.npz cora_v2_with_mask.npz citeseer_with_mask.npz reddit_with_mask.npz
do
  CUDA_VISIBLE_DEVICES=0 python vc_gcn_datanet.py --data-path  /home/dataset/$name --fuse true
  CUDA_VISIBLE_DEVICES=1 python vc_gat_datanet.py --data-path  /home/dataset/$name --fuse true
  CUDA_VISIBLE_DEVICES=2 python vc_appnp_datanet.py --data-path  /home/dataset/$name --fuse true
done
