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
import time
import argparse
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.context as context
from mindspore_gl import Graph, GraphField
from mindspore_gl.dataset import CoraV2
from mindspore_gl.nn import GNNCell
from mindspore_gl.sampling import negative_sample
from mindspore_gl.dataloader import split_data

from src.gae import GAENet, GCNEncoder, InnerProductDecoder
from util import get_auc_score


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
        predict = self.net(x, in_deg, out_deg, index, g)
        target = ops.Squeeze()(target)

        loss = self.loss_fn(predict, target)
        return loss

def get_pos_weight(node, pos):
    """
    Calculate the proportion of positive and negative samples.
    """
    pos_weight = float(node * node -(pos - node)) /(pos - node)
    return ms.Tensor((pos_weight,), ms.float32)

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
        context.set_context(device_target="GPU", save_graphs=True, save_graphs_path="./computational_graph/",
                            mode=context.GRAPH_MODE, enable_graph_kernel=True,
                            graph_kernel_flags="--enable_expand_ops=Gather --enable_cluster_ops=TensorScatterAdd,"
                                               "UnsortedSegmentSum, GatherNd --enable_recompute_fusion=false "
                                               "--enable_parallel_fusion=true ")
    else:
        context.set_context(device_target="GPU", mode=context.PYNATIVE_MODE)

    if args.profile:
        ms_profiler = Profiler(subgraph="ALL", is_detail=True, is_show_op_path=False, output_path="./prof_result")

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
    val_false = negative_sample(positive, ds.node_count-1, len(val), mode=mode)
    test_false = negative_sample(positive, ds.node_count-1, len(test), mode=mode)

    n_nodes = ds.node_feat.shape[0]
    n_edges = adj_coo.row.shape[0] - val.shape[0] - test.shape[0]
    in_deg = np.zeros(shape=n_nodes, dtype=np.int)
    out_deg = np.zeros(shape=n_nodes, dtype=np.int)

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
    encoder = GCNEncoder(data_feat_size=ds.num_features,
                         hidden_dim_size=(hidden1_dim, hidden2_dim),
                         activate=(ms.nn.ReLU(), None),
                         name='GAE')
    decoder = InnerProductDecoder(dropout_rate=dropout, decoder_type='all')
    net = GAENet(encoder,
                 decoder)
    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=lr, weight_decay=weight_decay)
    loss = LossNet(net, pos_weight)
    train_net = nn.TrainOneStepCell(loss, optimizer)

    index = 0
    for e in range(epochs):
        beg = time.time()
        train_net.set_train()

        loss_v = train_net(node_feat, in_deg, out_deg, label, index, *g.get_graph())
        end = time.time()
        dur = end - beg

        print("epoch:", e, "loss:", loss_v, "time:{} s".format(dur))

        net.set_train(False)
        out = net(node_feat, in_deg, out_deg, index, *g.get_graph())
        auc_score, ap_score = get_auc_score(out.asnumpy(), val, val_false)
        print('Val AUC score:', auc_score, "AP score:", ap_score, '\n')

    auc_score, ap_score = get_auc_score(out.asnumpy(), test, test_false)
    print('Test AUC score:', auc_score, "AP score:", ap_score)

    if args.profile:
        ms_profiler.analyse()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GCN for whole-graph classification')
    parser.add_argument("--data_name", type=str, default='cora_v2', choices=["cora_v2", "citeseer", "pubmed"])
    parser.add_argument("--data_path", type=str, default='./data/', help="path to dataset")
    parser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train(default: 200)')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate(default: 0.01)')
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--dropout_keep_prob", type=float, default=1.0, help="dropout")
    parser.add_argument("--hidden1_dim", type=int, default=32, help="num_hidden1")
    parser.add_argument("--hidden2_dim", type=int, default=16, help="num_hidden2")
    parser.add_argument("--mode", type=str, default="undirected", help="Sample matrix type")
    parser.add_argument('--profile', type=bool, default=False, help="feature dimension")
    parser.add_argument('--fuse', type=bool, default=False, help="enable fusion")
    parser.add_argument('--seed', type=int, default=42, help="seed")

    args = parser.parse_args()
    print("config:", args)
    main()
