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

from mindspore_gl import Graph
from mindspore_gl.dataset import Reddit
from mindspore_gl.nn import GNNCell
from mindspore_gl.graph import graph_csr_data

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
    device_id = args.device_id
    data_path = args.data_path
    context.set_context(device_target="GPU", save_graphs=False, save_graphs_path="./computational_graph/",
                        mode=context.GRAPH_MODE, enable_graph_kernel=True, device_id=device_id,
                        graph_kernel_flags="--enable_expand_ops=Gather --enable_cluster_ops=CSRReduceSum,CSRDiv "
                                           "--enable_recompute_fusion=false "
                                           "--enable_parallel_fusion=false "
                                           "--recompute_increment_threshold=40000000 "
                                           "--recompute_peak_threshold=3000000000 "
                                           "--enable_csr_fusion=true ")
    ds = Reddit(data_path)
    rerank = True
    csr_g, in_deg, out_deg, node_feat, node_label,\
    train_mask, _, test_mask = graph_csr_data(ds.adj_coo.row, ds.adj_coo.col, ds.node_count, ds.edge_count,
                                              node_feat=ds.node_feat, node_label=ds.node_label,
                                              train_mask=ds.train_mask, val_mask=ds.val_mask, test_mask=ds.test_mask,
                                              rerank=rerank)

    node_feat = ms.Tensor(node_feat, ms.float32)
    in_deg = ms.Tensor(in_deg, ms.int32)
    out_deg = ms.Tensor(out_deg, ms.int32)
    train_mask = ms.Tensor(train_mask, ms.float32)
    node_label = ms.Tensor(node_label, ms.int32)
    GNNCell.sparse_compute(csr=True, backward=True)
    net = GCNNet(data_feat_size=ds.node_feat_size,
                 hidden_dim_size=num_hidden,
                 n_classes=ds.num_classes,
                 dropout=dropout,
                 activation=ms.nn.ELU)
    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=lr, weight_decay=weight_decay)
    loss = LossNet(net)
    train_net = nn.TrainOneStepCell(loss, optimizer)
    total = 0.
    warm_up = 3
    for e in range(epochs):
        beg = time.time()
        train_net.set_train(True)
        train_net.set_grad()
        train_loss = train_net(node_feat, in_deg, out_deg, train_mask, node_label, csr_g[0], csr_g[1], csr_g[2],
                               csr_g[3], csr_g[4], csr_g[5])
        end = time.time()
        dur = end - beg
        if e >= warm_up:
            total = total + dur
        if test_mask is not None:
            net.set_train(False)
            out = net(node_feat, in_deg, out_deg, csr_g[0], csr_g[1], csr_g[2], csr_g[3], csr_g[4], csr_g[5]).asnumpy()
            predict = np.argmax(out[test_mask], axis=1)
            label = node_label.asnumpy()
            label = label[test_mask]
            count = np.equal(predict, label)
            print('Epoch time:{} ms Train loss {} Test acc:{}'.format(dur * 1000, train_loss,
                                                                      np.sum(count) / label.shape[0]))
        else:
            print('Epoch time:{} ms Train loss {}'.format(dur * 1000, train_loss))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GCN for whole-graph classification')
    parser.add_argument("--data_path", type=str, default='/home/dataset/', help="path to dataset")
    parser.add_argument("--device", type=str, default="GPU", help="which device to use")
    parser.add_argument("--device_id", type=int, default=0, help="which device id to use")
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout")
    parser.add_argument("--num_hidden", type=int, default=16, help="num_hidden")
    args = parser.parse_args()
    print(args)
    main()
