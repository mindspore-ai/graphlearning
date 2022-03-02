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
"""Utils"""
import json
from typing import Dict

from mindspore_gl.temp.graph_meta import MindsporeGlmeta


def load_meta(meta_file: str) -> MindsporeGlmeta:
    opened_file = open(meta_file, encoding="UTF-8")
    json_data: Dict = json.load(opened_file)
    meta_data = MindsporeGlmeta()
    meta_data.init_with_json(json_data)
    opened_file.close()
    return meta_data


def get_feat_info_from_meta(field_key: str, info: Dict):
    for feat_name in info:
        if info[feat_name]["name"] == field_key:
            return feat_name, info[feat_name]

    raise Exception(f"{field_key} doesn't exist")
