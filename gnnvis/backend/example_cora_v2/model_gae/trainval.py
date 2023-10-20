# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0(the "License");
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
"""train val"""
from gnnvis import GNNVis
import time
import argparse
import numpy as np
import os
import datetime
import json

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.context as context
from mindspore.profiler import Profiler
from mindspore_gl import Graph, GraphField
from mindspore_gl.dataset import CoraV2
from mindspore_gl.nn import GNNCell
from mindspore_gl.sampling import negative_sample
from mindspore_gl.dataloader import split_data
import pandas as pd

from src.gae import GAENet, GCNEncoder, InnerProductDecoder
from util import get_auc_score


import sys
sys.path.append("..")


def seed(x=2022):
    ms.set_seed(x)
    np.random.seed(x)


class LossNet(GNNCell):
    """
    Used to construct GAE Loss(BCELoss).

    ..math:
        L = E_{q(Z|X,A)}[log p(A | Z)]

    Args：
        net:GAEModel
        pos_weight(Tensor): Positive and negative sample ratio.
    """

    def __init__(self, net, pos_weight):
        super().__init__()
        self.net = net
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def construct(self, x, in_deg, out_deg, target, index, g: Graph):
        """
        Construct function for loss.

        Args：
            x(Tensor): The input node features.,shape:(node, feature_size)
            in_deg(Tensor): In degree, shape:(node)
            out_deg(Tensor): Out degree, shape:(node)
            target(Tensor): Adjacency Matrix Labels, shape:(node, node)
            g(Graph): The input graph.

        Returns:
            Tensor, output loss value.
        """
        # predict = self.net(x, in_deg, out_deg, index, g)
        # target = ops.Squeeze()(target)

        # loss = self.loss_fn(predict, target)

        # my
        predict_x, predict_y = self.net(x, in_deg, out_deg, index, g)
        target = ops.Squeeze()(target)

        loss = self.loss_fn(predict_y, target)
        # end my
        return loss


def get_pos_weight(node, pos):
    """
    Calculate the proportion of positive and negative samples.
    """
    pos_weight = float(node * node - (pos - node)) / (pos - node)
    return ms.Tensor((pos_weight,), ms.float32)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    """train gcn"""
    if seed is not None:
        seed(args.seed)

    dropout = args.dropout_keep_prob
    epochs = args.epochs
    hidden1_dim = args.hidden1_dim
    hidden2_dim = args.hidden2_dim
    lr = args.lr
    weight_decay = args.weight_decay
    mode = args.mode

    if args.fuse:
        context.set_context(device_target="CPU", save_graphs=True, save_graphs_path="./computational_graph/",
                            mode=context.GRAPH_MODE, enable_graph_kernel=True,
                            graph_kernel_flags="--enable_expand_ops=Gather --enable_cluster_ops=TensorScatterAdd,"
                                               "UnsortedSegmentSum, GatherNd --enable_recompute_fusion=false "
                                               "--enable_parallel_fusion=true ")
    else:
        # context.set_context(device_target="GPU", mode=context.GRAPH_MODE)
        context.set_context(device_target="CPU", mode=context.GRAPH_MODE)

    if args.profile:
        ms_profiler = Profiler(subgraph="ALL", is_detail=True,
                               is_show_op_path=False, output_path="./prof_result")

    # dataloader
    print(args.data_name)
    if args.data_name == 'cora_v2':
        ds = CoraV2(args.data_path, args.data_name)
    elif args.data_name == 'citeseer':
        ds = CoraV2(args.data_path, args.data_name)
    elif args.data_name == 'pubmed':
        ds = CoraV2(args.data_path, args.data_name)

    adj_coo, (train, val, test) = split_data(ds, graph_type='undirected')

    # Construct negative examples
    positive = [e for list in [train, val, test] for e in list]
    val_false = negative_sample(
        positive, ds.node_count - 1, len(val), mode=mode)
    test_false = negative_sample(
        positive, ds.node_count - 1, len(test), mode=mode)

    n_nodes = ds.node_feat.shape[0]
    n_edges = adj_coo.row.shape[0] - val.shape[0] - test.shape[0]
    in_deg = np.zeros(shape=n_nodes, dtype=np.int32)
    out_deg = np.zeros(shape=n_nodes, dtype=np.int32)

    # Construct labels, remove diagonal matrix
    label = ms.Tensor(adj_coo.toarray(), ms.float32)

    # Calculate in-degree and out-degree
    for r in adj_coo.row:
        out_deg[r] += 1
    for r in adj_coo.col:
        in_deg[r] += 1
    in_deg = ms.Tensor(in_deg, ms.int32)
    out_deg = ms.Tensor(out_deg, ms.int32)

    g = GraphField(ms.Tensor(adj_coo.row, dtype=ms.int32), ms.Tensor(adj_coo.col, dtype=ms.int32),
                   int(n_nodes), int(n_edges))
    node_feat = ms.Tensor(ds.node_feat, dtype=ms.float32)

    # pos_weight
    pos_weight = get_pos_weight(n_nodes, adj_coo.sum())

    # model and optimizer
    encoder = GCNEncoder(data_feat_size=ds.node_feat_size,
                         hidden_dim_size=(hidden1_dim, hidden2_dim),
                         activate=(ms.nn.ReLU(), None),
                         name='GAE')
    decoder = InnerProductDecoder(dropout_rate=dropout, decoder_type='all')
    net = GAENet(encoder,
                 decoder)
    optimizer = nn.optim.Adam(net.trainable_params(
    ), learning_rate=lr, weight_decay=weight_decay)
    loss = LossNet(net, pos_weight)
    train_net = nn.TrainOneStepCell(loss, optimizer)

    index = 0
    for e in range(epochs):
        beg = time.time()
        train_net.set_train()

        loss_v = train_net(node_feat, in_deg, out_deg,
                           label, index, *g.get_graph())
        end = time.time()
        dur = end - beg

        print("epoch:", e, "loss:", loss_v, "time:{} s".format(dur))

        net.set_train(False)
        # out = net(node_feat, in_deg, out_deg, index, *g.get_graph())
        # auc_score, ap_score = get_auc_score(out.asnumpy(), val, val_false)

        # my
        out_x, out_y = net(node_feat, in_deg, out_deg, index, *g.get_graph())
        auc_score, ap_score = get_auc_score(out_y.asnumpy(), val, val_false)
        # end my

        print('Val AUC score:', auc_score, "AP score:", ap_score, '\n')

    # auc_score, ap_score = get_auc_score(out.asnumpy(), test, test_false)

    # my
    auc_score, ap_score = get_auc_score(out_y.asnumpy(), test, test_false)
    # end my

    print('Test AUC score:', auc_score, "AP score:", ap_score)

    # 保存
    matrix = out_y.asnumpy()
    print("len(positive):", len(positive))
    preds_bool_index = []
    for e in positive:
        preds_bool_index.append(sigmoid(matrix[e[0], e[1]]) > 0.5)

    for e in positive:
        if (matrix[e[0], e[1]] < 0):
            print('a negative', e)

    print("pres_bool_index OK")
    print("np.all(preds_bool_index): ", np.all(preds_bool_index))

    positive = np.array(positive, dtype=np.int32)
    true_allow = positive[preds_bool_index]
    print(true_allow[0:20])
    preds_bool_index = [not i for i in preds_bool_index]
    false_allow = positive[preds_bool_index]
    print(false_allow[0:20])
    print("true allow, false allow OK")

    positive = positive.tolist()

    trueUnseenEdgesSorted = []
    for row, vec in enumerate(matrix.tolist()):
        if row % 100 == 0:
            print(f"row:{row} now processing")

        vec_index = [(col, data) for col, data in enumerate(vec)]
        sorted_vec = sorted(vec_index, key=lambda x: x[1], reverse=True)

        if row % 100 == 0:
            # print('sorted_vec', sorted_vec[:10])
            print(f"row:{row} sorted")

        count, i = 0, 0
        top_k_list = []
        while count < args.top_k:
            if [row, sorted_vec[i][0]] not in positive:
                # print(row,sorted_vec[i][0]," not in")
                top_k_list.append(sorted_vec[i])
                count += 1
            i += 1
            if i == len(sorted_vec):
                break

        if count > 0:
            if row % 100 == 0:
                print('top_k_list:', top_k_list)
            trueUnseenEdgesSorted.append([row, top_k_list])

        # if (row+1)%200 ==0:
        #     print(f"row:{row} OK")
        # break

    # 再按top1排序, 因为对称性,肯定是top1相等的成对存在, 所以让index小的在前
    # 1是说top_k_list 在[row, top_k_list], 0是说最大的那个, 再取data或者index
    trueUnseenEdgesSorted = sorted(trueUnseenEdgesSorted, key=lambda x: (
        x[1][0][1], x[1][0][0]), reverse=True)
    trueUnseenEdgesSortedCompact = dict(
        [[i[0], [j[0] for j in i[1]]] for i in trueUnseenEdgesSorted])

    save_json = {
        "isLinkPrediction": True,
        "trueAllowEdges": true_allow.tolist(),
        "falseAllowEdges": false_allow.tolist(),
        "trueUnseenEdgesSorted": trueUnseenEdgesSortedCompact,
    }

    # 创建文件夹
    timestamp = datetime.datetime.strftime(
        datetime.datetime.now(), '%Y-%m-%d %H:%M:%S ')
    save_path = f'./saved_out_gae_link_pred/{args.data_name}/{timestamp}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # with open(os.path.join(save_path, "prediction-results.json"), mode='w') as f_pred:
    #     json.dump(save_json, f_pred, ensure_ascii=False)
    # f_pred.close()

    # df = pd.DataFrame(out_x.asnumpy(), index=None, columns=None)
    # df.to_csv(os.path.join(save_path, 'node-embeddings.csv'), header=None, index=None)

    # df_test = pd.DataFrame(test,index=None, columns=None)
    # df_test.to_csv(f"./link_pred/{args.data_name}_test.csv", header=None,index=None)
    # df_test_false = pd.DataFrame(test_false,index=None, columns=None)
    # df_test_false.to_csv(f"./link_pred/{args.data_name}_test_false.csv", header=None,index=None)

    # 保存
    # end my

    # ms.export(net, node_feat, in_deg, out_deg, index, *g.get_graph(), file_name="gae_model", file_format="MINDIR")
    #
    # if args.profile:
    #     ms_profiler.analyse()

    G = None
    with open(f"./{args.data_path}/graph.json", mode='r') as f:
        G = json.load(f)

    GNNVis(G,
           node_embed=out_x,
           node_dense_features=ds.node_feat,
           link_pred_res=save_json,
           gen_path=save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='GAE for link prediction')
    parser.add_argument("--data_name", type=str, default='cora_v2')
    parser.add_argument("--data_path", type=str,
                        default='./dataset_cora_v2/', help="path to dataset")
    parser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train(default: 200)')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate(default: 0.01)')
    parser.add_argument("--weight_decay", type=float,
                        default=0.0, help="weight decay")
    parser.add_argument("--dropout_keep_prob", type=float,
                        default=1.0, help="dropout")
    parser.add_argument("--hidden1_dim", type=int,
                        default=32, help="num_hidden1")
    parser.add_argument("--hidden2_dim", type=int,
                        default=16, help="num_hidden2")
    parser.add_argument("--mode", type=str,
                        default="undirected", help="Sample matrix type")
    parser.add_argument('--profile', type=bool,
                        default=False, help="feature dimension")
    parser.add_argument('--fuse', type=bool, default=False,
                        help="enable fusion")
    parser.add_argument('--seed', type=int, default=42, help="seed")
    parser.add_argument('--top_k', type=int, default=5,
                        help='top-k of the recommended links (not exist in origin dataset)')

    args = parser.parse_args()
    print("config:", args)
    main()
