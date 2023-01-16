# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License")
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
# ==============================================================================
"""
Graph dataloader convert tool for MindRecord.
"""
from typing import List, Dict

import numpy as np
from mindspore import log as logger
from mindspore_gl.temp import MindsporeGlmeta, MindRecordDatatype, MindRecordDataShape

__all__ = ['GraphMapSchema']


class GraphMapSchema:
    """
    Class is for transformation from graph dataloader to MindRecord.
    """

    def __init__(self):
        """
        init
        """
        self.num_node_features = 0
        self.edge_feat_size = 0
        self.union_schema_in_mindrecord = {
            "first_id": {"type": "int64"},
            "second_id": {"type": "int64"},
            "third_id": {"type": "int64"},
            "type": {"type": "int32"},
            "weight": {"type": "float32"},
            "attribute": {"type": "string"},  # 'n' for ndoe, 'e' for edge
            "node_feature_index": {"type": "int32", "shape": [-1]},
            "edge_feature_index": {"type": "int32", "shape": [-1]}
        }
        self.meta = MindsporeGlmeta()

    @property
    def schema(self):
        """
        Get schema
        """
        return self.union_schema_in_mindrecord

    @property
    def graph_meta(self):
        return self.meta

    @staticmethod
    def _valid(feature_names: List[str], feature_datatypes: List[MindRecordDatatype],
               feature_shapes: List[MindRecordDataShape]):
        assert len(feature_names) == len(
            feature_datatypes), "feature names length should match feature datatypes length"
        assert len(feature_names) == len(feature_shapes), "feature names length should match feature shapes length"

    def add_node_info(self, node_type_names: List[str], node_types: List[int]):
        pass

    def add_edge_info(self, edge_type_names: List[str], edge_types: List[int]):
        pass

    def add_node_feature_schema(self, feature_name: str, feature_datatype: MindRecordDatatype,
                                feature_shape: MindRecordDataShape):
        """
        add node feature schema function
        """
        self.add_node_features_schema([feature_name], [feature_datatype], [feature_shape])

    def add_node_features_schema(self, feature_names: List[str], feature_datatypes: List[MindRecordDatatype],
                                 feature_shapes: List[MindRecordDataShape]):
        """
        add node features schema function
        """
        self._valid(feature_names, feature_datatypes, feature_shapes)
        for name, data_type, shape in zip(feature_names, feature_datatypes, feature_shapes):
            field_key = f"node_feature_{self.num_node_features + 1}"
            field_value = {"type": data_type.value[0], "shape": shape.shape}
            self.union_schema_in_mindrecord[field_key] = field_value
            meta_value = {"name": field_key}
            meta_value.update(field_value)
            self.meta.node_feat_info[name] = meta_value
            self.num_node_features += 1

    def add_edge_feature_schema(self, feature_name: str, feature_datatype: MindRecordDatatype,
                                feature_shape: MindRecordDataShape):
        """
        add edge feature schema function
        """
        self.add_edge_features_schema([feature_name], [feature_datatype], [feature_shape])

    def add_edge_features_schema(self, feature_names: List[str], feature_datatypes: List[MindRecordDatatype],
                                 feature_shapes: List[MindRecordDataShape]):
        """
        add edge features schema function
        """

        self._valid(feature_names, feature_datatypes, feature_shapes)

        for name, data_type, shape in zip(feature_names, feature_datatypes, feature_shapes):
            field_key = f"edge_feature_{self.edge_feat_size + 1}"
            field_value = {"type": data_type.value[0], "shape": shape.shape}
            self.union_schema_in_mindrecord[field_key] = field_value
            meta_value = {"name": field_key}
            meta_value.update(field_value)
            self.meta.edge_feat_info[name] = meta_value
            self.edge_feat_size += 1

    def add_meta_variants(self, meta: Dict):
        """
        add varants dataloader
        """
        for key, value in meta.items():
            self.meta.variants[key] = value

    def transform_node(self, node):
        """
        Executes transformation from node dataloader to union format.

        Args:
            node(schema): node's dataloader.

        Returns:
            graph dataloader with union schema.
        """
        if node is None:
            logger.info("node cannot be None.")
            raise ValueError("node cannot be None.")

        node_graph = {"first_id": node["id"], "second_id": 0, "third_id": 0, "weight": 1.0, "attribute": 'n',
                      "type": node["type"], "node_feature_index": []}
        if "weight" in node:
            node_graph["weight"] = node["weight"]

        for i in range(self.num_node_features):
            k = i + 1
            node_field_key = 'feature_' + str(k)
            graph_field_key = 'node_feature_' + str(k)
            graph_field_type = self.union_schema_in_mindrecord[graph_field_key]["type"]
            if node_field_key in node:
                node_graph["node_feature_index"].append(k)
                node_graph[graph_field_key] = np.reshape(np.array(node[node_field_key], dtype=graph_field_type), [-1])
            else:
                node_graph[graph_field_key] = np.reshape(np.array([0], dtype=graph_field_type), [-1])

        if node_graph["node_feature_index"]:
            node_graph["node_feature_index"] = np.array(node_graph["node_feature_index"], dtype="int32")
        else:
            node_graph["node_feature_index"] = np.array([-1], dtype="int32")

        node_graph["edge_feature_index"] = np.array([-1], dtype="int32")
        for i in range(self.edge_feat_size):
            k = i + 1
            graph_field_key = 'edge_feature_' + str(k)
            graph_field_type = self.union_schema_in_mindrecord[graph_field_key]["type"]
            node_graph[graph_field_key] = np.reshape(np.array([0], dtype=graph_field_type), [-1])
        return node_graph

    def transform_edge(self, edge):
        """
        Executes transformation from edge dataloader to union format.

        Args:
            edge(schema): edge's dataloader.

        Returns:
            graph dataloader with union schema.
        """
        if edge is None:
            logger.info("edge cannot be None.")
            raise ValueError("edge cannot be None.")

        edge_graph = {"first_id": edge["id"], "second_id": edge["src_id"], "third_id": edge["dst_id"], "weight": 1.0,
                      "attribute": 'e', "type": edge["type"], "edge_feature_index": []}

        if "weight" in edge:
            edge_graph["weight"] = edge["weight"]

        for i in range(self.edge_feat_size):
            k = i + 1
            edge_field_key = 'feature_' + str(k)
            graph_field_key = 'edge_feature_' + str(k)
            graph_field_type = self.union_schema_in_mindrecord[graph_field_key]["type"]
            if edge_field_key in edge:
                edge_graph["edge_feature_index"].append(k)
                edge_graph[graph_field_key] = np.reshape(np.array(edge[edge_field_key], dtype=graph_field_type), [-1])
            else:
                edge_graph[graph_field_key] = np.reshape(np.array([0], dtype=graph_field_type), [-1])

        if edge_graph["edge_feature_index"]:
            edge_graph["edge_feature_index"] = np.array(edge_graph["edge_feature_index"], dtype="int32")
        else:
            edge_graph["edge_feature_index"] = np.array([-1], dtype="int32")

        edge_graph["node_feature_index"] = np.array([-1], dtype="int32")
        for i in range(self.num_node_features):
            k = i + 1
            graph_field_key = 'node_feature_' + str(k)
            graph_field_type = self.union_schema_in_mindrecord[graph_field_key]["type"]
            edge_graph[graph_field_key] = np.array([0], dtype=graph_field_type)
        return edge_graph
