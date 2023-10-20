import os
import unittest
from backend import GNNVis
from example_data import graph_dict, node_embed, graph_embed, graph_classify_res


class TestGnnVisGraphClassification(unittest.TestCase):
    save_path = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = TestGnnVisGraphClassification.save_path
        self.save_path = os.path.join(self.save_path, "TestGnnVisGraphClassification")
        self.init()

    def init(self):
        self.gnn_vis = GNNVis(
            graph=graph_dict,
            node_embed=node_embed,
            graph_embed=graph_embed,
            graph_classify_res=graph_classify_res,
            gen_path=self.save_path
        )

        self.file_list = [
            "graph.json",
            "initial-layout.json",
            "node-embeddings.csv",
            "graph-embeddings.csv",
            "prediction-results.json",
        ]

    def test_file_exist(self):
        for file_name in self.file_list:
            file_path = os.path.join(self.save_path, file_name)
            self.assertTrue(os.path.exists(file_path))
