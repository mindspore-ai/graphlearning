import os
import json

import numpy as np

import networkx as nx

import umap


class JsonEncoder(json.JSONEncoder):
    """Convert numpy classes to JSON serializable objects."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)


class Utils:
    """工具类"""

    @staticmethod
    def export_dict_to_json(dict_data: dict, file_path: str, file_name: str):
        """将字典数据导出为 json 文件并保存"""
        res = json.dumps(dict_data, separators=(',', ':'), ensure_ascii=False, cls=JsonEncoder)

        path = os.path.join(file_path, file_name)
        with open(path, 'w') as f:
            f.write(res)

    @staticmethod
    def export_numpy_to_csv(numpy_data: np.ndarray, file_path: str, file_name: str, title: list = None):
        """将 numpy 数据导出为 csv 文件并保存"""
        path = os.path.join(file_path, file_name)

        args = {'delimiter': ',',
                'fmt': '%.3f',
                "header": ','.join(title) if title is not None else "",
                }
        if title is not None:
            args["comments"] = ""

        np.savetxt(path, numpy_data, **args)


class GNNVis:
    def __init__(
            self,
            graph: dict,
            node_embed: np.ndarray,
            node_dense_features: np.ndarray = None,
            node_dense_features_name: list = None,
            node_sparse_features: dict = None,
            link_pred_res: dict = None,
            node_classify_res: dict = None,
            graph_classify_res: dict = None,
            graph_embed: np.ndarray = None,
            gen_path: str = None,
    ):
        self.graph = graph

        self.node_embed = node_embed
        self.node_dense_features = node_dense_features
        self.node_dense_features_name = node_dense_features_name
        self.node_sparse_features = node_sparse_features

        self.link_pred_res = link_pred_res
        self.node_classify_res = node_classify_res
        self.graph_classify_res = graph_classify_res

        self.graph_embed = graph_embed
        self.gen_path = gen_path if gen_path else r'./'

        print(f"Running GNNVis...")

        self._check_gen_path()
        self._save_graph()

        if self.node_embed is not None:
            self._save_node_embed()

        if self.node_dense_features is not None:
            self._save_node_dense_features()

        if self.node_sparse_features is not None:
            self._save_node_sparse_features()

        if self.link_pred_res is not None or \
                self.node_classify_res is not None or \
                self.graph_classify_res is not None:
            self._save_predict_res()

        if self.graph_embed is not None:
            self._save_graph_embed()

        if self.node_embed is not None:
            self._gen_initial_layout()

    def _check_gen_path(self):
        """检查生成路径是否存在，若不存在则创建"""
        if not os.path.exists(self.gen_path):
            os.makedirs(self.gen_path)

    def _save_graph(self):
        """将 json 格式的图数据计算一阶邻点后保存"""
        print("Saving graph...")
        g = nx.node_link_graph(self.graph, link="edges")

        # 计算一阶邻点及关联边
        edge_dict = []
        for current_id in list(g.nodes):
            one_neighbor = []

            ne_list = list(g.neighbors(current_id))
            for ne in ne_list:
                if g.is_multigraph():
                    edges = g.get_edge_data(current_id, ne)
                    for edge_info in edges.values():
                        one_neighbor.append({"nid": ne, "eid": edge_info['eid']})
                else:
                    edge = g.get_edge_data(current_id, ne)
                    one_neighbor.append({"nid": ne, "eid": edge['eid']})

            edge_dict.append(one_neighbor)

        res = nx.node_link_data(g, link="edges")
        if "graph" in res:
            res["graphs"] = res["graph"]
            del res["graph"]
        res["edgeDict"] = edge_dict

        Utils.export_dict_to_json(res, self.gen_path, 'graph.json')

        self.g = g

    def _save_node_embed(self):
        """将节点嵌入保存为 csv 文件"""
        Utils.export_numpy_to_csv(self.node_embed, self.gen_path, 'node-embeddings.csv')

    def _save_graph_embed(self):
        """将图嵌入保存为 csv 文件"""
        Utils.export_numpy_to_csv(self.graph_embed, self.gen_path, 'graph-embeddings.csv')

    def _save_node_dense_features(self):
        """将节点密集特征保存为 csv 文件"""
        title = None
        if self.node_dense_features_name is not None:
            title = [f"feat_{i}" for i in range(len(self.node_dense_features[0]))]

        Utils.export_numpy_to_csv(self.node_dense_features, self.gen_path, 'node-dense-features.csv',
                                  title=title)

    def _save_node_sparse_features(self):
        """将节点稀疏特征保存为 csv 文件"""
        Utils.export_dict_to_json(self.node_sparse_features, self.gen_path, 'node-sparse-features.json')

    def _save_predict_res(self):
        """将预测结果保存为 json 文件"""
        print("Saving prediction results...")
        res = {}

        if self.link_pred_res:
            res["taskType"] = "link-prediction"
            res.update(self.link_pred_res)
        elif self.node_classify_res:
            res["taskType"] = "node-classification"
            res.update(self.node_classify_res)
        elif self.graph_classify_res:
            res["taskType"] = "graph-classification"
            res.update(self.graph_classify_res)

        Utils.export_dict_to_json(res, self.gen_path, 'prediction-results.json')

    def _gen_initial_layout(self):
        """生成图中节点的初始布局"""
        print(f"Generating initial layout...")

        res = {}

        force_layout_dict = nx.spring_layout(self.g, scale=100)
        force_layout = []
        for nid, (x, y) in force_layout_dict.items():
            force_layout.append({"id": nid, "x": x, "y": y})
        res["forceDirectedLayout"] = force_layout

        reducer = umap.UMAP()
        node_embed_2d = reducer.fit_transform(self.node_embed)
        res["nodeEmbUmp"] = node_embed_2d.tolist()

        if self.graph_embed is not None:
            reducer = umap.UMAP()
            graph_embed_2d = reducer.fit_transform(self.graph_embed)
            res["graphEmbUmp"] = graph_embed_2d.tolist()

        Utils.export_dict_to_json(res, self.gen_path, 'initial-layout.json')
