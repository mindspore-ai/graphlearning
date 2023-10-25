import json
import os

import networkx as nx
import numpy as np

# 可选任务列表
TASK_TYPE_LIST = ["node-classification", "link-prediction", "graph-classification"]

# 为 点/边/图 生成指标
INDEX_TARGET_LIST = ["node", "edge", "graph"]


# 【需要用户自定义】配置参数
class Attr:
    def __init__(self):
        # 预测任务类型，可选项列表为 TASK_TYPE_LIST
        self.task_type = "graph-classification"

        # 生成指标的对象，可选项列表为 INDEX_TARGET_LIST
        self.index_target = "graph"

        # 数据文件路径
        self.data_path = "../../datasets/saved_out_mpnn"

        # 保存文件名后缀（例如：当针对图计算指标时，保存文件名为 "graph-custom-index.json"）
        self.save_file_name_suffix = "custom-index.json"


# 保存用户读取的数据
class Data:
    def __init__(self):
        # NetworkX 格式的图数据
        self.g = None

        # embedding，如果没有则为 None
        self.embedding = None

        # 预测结果，如果没有则为 None
        self.prediction_result = None

    def __repr__(self):
        return f"Data(g, embedding={self.embedding}, prediction_result={self.prediction_result})"


attr = Attr()
data = Data()


def check_config(gl_attr=None):
    """检查配置参数是否合法"""
    if gl_attr is None:
        gl_attr = attr

    if gl_attr.task_type not in TASK_TYPE_LIST:
        raise ValueError(f"Invalid task type, please choose ONE from {TASK_TYPE_LIST}.")

    if gl_attr.index_target not in INDEX_TARGET_LIST:
        raise ValueError(f"Invalid index target, please choose ONE from {INDEX_TARGET_LIST}.")

    return True


def read_data(gl_attr=None, gl_data=None):
    """
    从 json 文件读取数据
    """
    if gl_attr is None:
        gl_attr = attr
    if gl_data is None:
        gl_data = data

    # 读取图数据
    graph_json_path = os.path.join(gl_attr.data_path, "graph.json")
    graph_json = json.load(open(graph_json_path))
    if "graphs" in graph_json:
        graph_json["graph"] = graph_json["graphs"]
        del graph_json["graphs"]
    nx_graph = nx.node_link_graph(graph_json, link="edges")
    gl_data.g = nx_graph

    # 读取 embedding
    embedding_path = os.path.join(gl_attr.data_path, f"{gl_attr.index_target}-embeddings.csv")
    if os.path.isfile(embedding_path):
        embedding = np.loadtxt(embedding_path, delimiter=',')
        gl_data.embedding = np.array(embedding)
    else:
        print(f"Skip reading [{gl_attr.index_target}-embeddings.csv] ...")

    # 读取预测结果
    prediction_result_path = os.path.join(gl_attr.data_path, "prediction-results.json")
    if os.path.isfile(prediction_result_path):
        prediction_result = json.load(open(prediction_result_path))
        gl_data.prediction_result = prediction_result
    else:
        print(f"Skip reading [prediction-results.json] ...")


def get_count_of_elements(numerator_graph, element_id, gl_data=None):
    """
    【此函数需要用户自定义】[Dataset: Mutag] 计算单个分子中指定编号的原子数
    :param numerator_graph: 指定分子的图数据
    :param element_id: 指定原子编号
    :param gl_data: 数据存放点
    :return: 原子数
    """
    if gl_data is None:
        gl_data = data

    cnt = 0
    for nid in numerator_graph["nodes"]:
        if gl_data.g.nodes[nid]["label"] == element_id:
            cnt += 1
    return cnt


def get_compute_count_of_NH2(numerator_graph, gl_data=None):
    """
    [Dataset: Mutag] 计算单个分子中「氨基」的数目
    :param numerator_graph: 指定分子的图数据
    :param gl_data: 数据存放点
    :return: NH2 的数目
    """
    if gl_data is None:
        gl_data = data

    cnt = 0
    for nid in numerator_graph["nodes"]:
        if gl_data.g.nodes[nid]["label"] != 1:
            continue
        if sum(gl_data.g.nodes[neighbor_id]["label"] == 2 for neighbor_id in gl_data.g.neighbors(nid)) >= 2:
            cnt += 1
    return cnt


def calc_index(sub_graph, gl_data=None):
    """【此函数需要用户自定义】对每张图计算指标"""

    if gl_data is None:
        gl_data = data

    return {
        "number_of_C": get_count_of_elements(sub_graph, 'C', gl_data),
        "number_of_F": get_count_of_elements(sub_graph, 'F', gl_data),
        "number_of_NH2": get_compute_count_of_NH2(sub_graph, gl_data),
    }


def get_custom_index(gl_attr=None, gl_data=None):
    if gl_attr is None:
        gl_attr = attr
    if gl_data is None:
        gl_data = data

    res = {"index_target": gl_attr.index_target}

    # 为每张图计算指标
    for sub_graph_id, sub_graph in gl_data.g.graph.items():
        sub_graph_custom_index = calc_index(sub_graph, gl_data)
        for k, v in sub_graph_custom_index.items():
            if k not in res:
                res[k] = {}
            res[k][sub_graph_id] = v

    return res


def main():
    read_data()
    res = get_custom_index()

    # 保存指标到 json 文件
    file_name = f"{attr.index_target}-{attr.save_file_name_suffix}"
    res = json.dumps(res, separators=(',', ':'), ensure_ascii=False)

    path = os.path.join(attr.data_path, file_name)
    with open(path, 'w') as f:
        f.write(res)

    print("Saved custom index.")


if __name__ == "__main__":
    check_config()
    main()
