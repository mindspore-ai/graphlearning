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

"""Graph metat data information"""
from typing import Dict, List
from enum import Enum

__all__ = [
    "MindRecordDatatype",
    "MindRecordDataShape",
    "DEFAULT_DATA_SHAPE",
    "MindsporeGlmeta"
]


class MindRecordDatatype(Enum):
    UINT8 = "uint8"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    STRING = "string"


class MindRecordDataShape:

    def __init__(self, shape: List[int]):
        self.shape = shape


DEFAULT_DATA_SHAPE = MindRecordDataShape([-1])


class MindsporeGlmeta:
    """GL Meta"""

    def __init__(self):

        self._graph_info: Dict = {}
        self._node_feat_info: Dict = {}
        self._edge_feat_info: Dict = {}
        self._variants: Dict = {}

        self._keys = ["graph_info", "node_feat_info", "edge_feat_info", "variants"]

    @property
    def json(self):
        return {
            "graph_info": self._graph_info,
            "node_feat_info": self._node_feat_info,
            "edge_feat_info": self._edge_feat_info,
            "variants": self._variants
        }

    def init_with_json(self, json_data: Dict):
        for key in self._keys:
            assert json_data.get(key, None) is not None, f"{key} missing in meta data"
        self._graph_info = json_data["graph_info"]
        self._node_feat_info = json_data["node_feat_info"]
        self._edge_feat_info = json_data["edge_feat_info"]
        self._variants = json_data["variants"]

    @property
    def graph_info(self):
        return self._graph_info

    @property
    def node_feat_info(self):
        return self._node_feat_info

    @property
    def edge_feat_info(self):
        return self._edge_feat_info

    @property
    def variants(self):
        return self._variants
