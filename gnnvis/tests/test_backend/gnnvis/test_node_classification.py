import os
import unittest
from backend import GNNVis
from example_data import graph_dict, node_embed, node_classify_res, node_dense_features, node_dense_features_name


class TestGnnVisNodeClassification(unittest.TestCase):
    save_path = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = TestGnnVisNodeClassification.save_path
        self.save_path = os.path.join(self.save_path, "TestGnnVisNodeClassification")
        self.init()

    def init(self):
        self.gnn_vis = GNNVis(
            graph=graph_dict,
            node_embed=node_embed,
            node_dense_features=node_dense_features,
            node_dense_features_name=node_dense_features_name,
            node_classify_res=node_classify_res,
            gen_path=self.save_path
        )

        self.file_list = [
            "graph.json",
            "initial-layout.json",
            "node-embeddings.csv",
            "node-dense-features.csv",
            "prediction-results.json",
        ]

    def test_file_exist(self):
        for file_name in self.file_list:
            file_path = os.path.join(self.save_path, file_name)
            self.assertTrue(os.path.exists(file_path))
