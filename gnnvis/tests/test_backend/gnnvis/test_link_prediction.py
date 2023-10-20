import os
import unittest
from backend import GNNVis
from example_data import graph_dict, node_embed, node_sparse_features, link_pred_res


class TestGnnVisLinkPrediction(unittest.TestCase):
    save_path = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = TestGnnVisLinkPrediction.save_path
        self.save_path = os.path.join(self.save_path, "TestGnnVisLinkPrediction")
        self.init()

    def init(self):
        self.gnn_vis = GNNVis(
            graph=graph_dict,
            node_embed=node_embed,
            link_pred_res=link_pred_res,
            node_sparse_features=node_sparse_features,
            gen_path=self.save_path
        )

        self.file_list = [
            "graph.json",
            "initial-layout.json",
            "node-embeddings.csv",
            "node-sparse-features.json",
            "prediction-results.json",
        ]

    def test_file_exist(self):
        for file_name in self.file_list:
            file_path = os.path.join(self.save_path, file_name)
            self.assertTrue(os.path.exists(file_path))
