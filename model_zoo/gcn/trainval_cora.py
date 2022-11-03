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

from src.gcn import GCNNet


class LossNet(GNNCell):
    """ LossNet definition """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')

    def construct(self, x, in_deg, out_deg, train_mask, target, g: Graph):
        """GCN Net with loss function"""
        predict = self.net(x, in_deg, out_deg, g)
        target = ops.Squeeze()(target)
        loss = self.loss_fn(predict, target)
        loss = loss * train_mask
        return ms.ops.ReduceSum()(loss) / ms.ops.ReduceSum()(train_mask)


def main():
    """train gcn"""

    dropout = args.dropout
    epochs = args.epochs
    num_hidden = args.num_hidden
    lr = args.lr
    weight_decay = args.weight_decay

    context.set_context(device_target="GPU", mode=context.GRAPH_MODE, enable_graph_kernel=True)
    # dataloader
    ds = CoraV2(args.data_path)

    n_nodes = ds.node_feat.shape[0]
    n_edges = ds.adj_coo.row.shape[0]
    in_deg = np.zeros(shape=n_nodes, dtype=np.int)
    out_deg = np.zeros(shape=n_nodes, dtype=np.int)
    for r in ds.adj_coo.row:
        out_deg[r] += 1
    for r in ds.adj_coo.col:
        in_deg[r] += 1
    in_deg = ms.Tensor(in_deg, ms.int32)
    out_deg = ms.Tensor(out_deg, ms.int32)
    g = GraphField(ms.Tensor(ds.adj_coo.row, dtype=ms.int32), ms.Tensor(ds.adj_coo.col, dtype=ms.int32),
                   int(n_nodes), int(n_edges))
    node_feat = ms.Tensor(ds.node_feat)
    train_mask = ms.Tensor(ds.train_mask, ms.float32)
    node_label = ms.Tensor(ds.node_label)

    # model
    net = GCNNet(data_feat_size=ds.num_features,
                 hidden_dim_size=num_hidden,
                 n_classes=ds.num_classes,
                 dropout=dropout,
                 activation=ms.nn.ELU)
    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=lr, weight_decay=weight_decay)
    loss = LossNet(net)
    train_net = nn.TrainOneStepCell(loss, optimizer)
    total = 0.
    warm_up = 3
    best_acc = 0.

    for e in range(epochs):
        beg = time.time()
        train_net.set_train()
        train_net(node_feat, in_deg, out_deg, train_mask, node_label, *g.get_graph())
        end = time.time()
        dur = end - beg
        if e >= warm_up:
            total = total + dur

        net.set_train(False)
        out = net(node_feat, in_deg, out_deg, *g.get_graph()).asnumpy()
        test_mask = ds.test_mask
        labels = ds.node_label
        predict = np.argmax(out[test_mask], axis=1)
        label = labels[test_mask]
        count = np.equal(predict, label)
        test_acc = np.sum(count) / label.shape[0]
        best_acc = test_acc if test_acc > best_acc else best_acc
        print('epoch:', e, ' test_acc:', test_acc)
    print('best_acc:', best_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GCN for whole-graph classification')
    parser.add_argument("--data_path", type=str, default='/home/dataset/', help="path to dataset")
    parser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout")
    parser.add_argument("--num_hidden", type=int, default=16, help="num_hidden")
    args = parser.parse_args()
    print(args)
    main()
