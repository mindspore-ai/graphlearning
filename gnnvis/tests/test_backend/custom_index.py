import os.path
import json
import unittest
import networkx as nx
import numpy as np

from backend.custom_index import Attr, Data, check_config, read_data, get_custom_index
from example_data import graph_dict, graph_embed, graph_classify_res


class TestCustomIndex(unittest.TestCase):
    save_path = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join(TestCustomIndex.save_path, "TestCustomIndex")
        self.init()
        self.prepare_data()
        read_data(self.attr, self.data)
        self.res = get_custom_index(self.attr, self.data)

    def init(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.attr = Attr()
        self.data = Data()

        self.attr.task_type = "graph-classification"
        self.attr.index_target = "graph"
        self.attr.data_path = self.save_path

    def test_check_config(self):
        self.assertTrue(check_config(self.attr))

    def prepare_data(self):
        for json_data in [
            [graph_dict, "graph.json"],
            [graph_classify_res, "prediction-results.json"],
        ]:
            res = json.dumps(json_data[0], separators=(',', ':'), ensure_ascii=False)
            path = os.path.join(self.save_path, json_data[1])
            with open(path, 'w') as f:
                f.write(res)

        path = os.path.join(self.save_path, "graph-embeddings.csv")
        np.savetxt(path, graph_embed, delimiter=',')

    def test_graph(self):
        graph = nx.node_link_data(self.data.g, link="edges")
        if "graph" in graph:
            graph["graphs"] = graph["graph"]
            del graph["graph"]
        self.assertEqual(graph["graphs"], graph_dict["graphs"])
        self.assertEqual(graph["nodes"], graph_dict["nodes"])

    def test_embedding(self):
        self.assertTrue((self.data.embedding == graph_embed).all())

    def test_prediction_result(self):
        self.assertEqual(self.data.prediction_result, graph_classify_res)

    def test_res(self):
        self.assertEqual(self.res["index_target"], "graph")
        self.assertEqual(self.res["number_of_C"], {"0": 1, "1": 0})
        self.assertEqual(self.res["number_of_F"], {"0": 0, "1": 1})
        self.assertEqual(self.res["number_of_NH2"], {"0": 0, "1": 0})
