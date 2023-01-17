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
from mindspore.train.callback import TimeMonitor, LossMonitor

from mindspore_gl import Graph, GraphField
from mindspore_gl.dataset import CoraV2
from mindspore_gl.nn import GNNCell

from src.gatv2 import GatV2Net


class LossNet(GNNCell):
    """ LossNet definition """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')

    def construct(self, x, target, train_mask, g: Graph):
        """GATv2 Net with loss function"""
        predict = self.net(x, g)
        target = ops.Squeeze()(target)
        loss = self.loss_fn(predict, target)
        loss = loss * train_mask
        return ms.ops.ReduceSum()(loss) / ms.ops.ReduceSum()(train_mask)


def main():
    """train gatv2"""
    context.set_context(device_target=args.device, mode=context.GRAPH_MODE, enable_graph_kernel=True,
                        device_id=args.device_id)
    num_layers = args.num_layers
    num_hidden = args.num_hidden
    num_heads = args.num_heads
    num_out_heads = args.num_out_heads
    in_drop = args.in_drop
    attn_drop = args.attn_drop
    negative_slope = args.negative_slope
    lr = args.lr
    weight_decay = args.weight_decay
    epochs = args.epochs

    # dataloader
    ds = CoraV2(args.data_path)

    n_nodes = ds.node_feat.shape[0]
    n_edges = ds.adj_coo.row.shape[0]
    g = GraphField(ms.Tensor(ds.adj_coo.row, dtype=ms.int32), ms.Tensor(ds.adj_coo.col, dtype=ms.int32),
                   int(n_nodes), int(n_edges))
    node_feat = ms.Tensor(ds.node_feat)
    train_mask = ms.Tensor(ds.train_mask, ms.float32)
    node_label = ms.Tensor(ds.node_label)

    TimeMonitor()
    LossMonitor()

    # model
    net = GatV2Net(num_layers=num_layers,
                   data_feat_size=ds.node_feat_size,
                   hidden_dim_size=num_hidden,
                   n_classes=ds.num_classes,
                   heads=[num_heads for _ in range(num_layers)] + [num_out_heads],
                   input_drop_out_rate=in_drop,
                   attn_drop_out_rate=attn_drop,
                   leaky_relu_slope=negative_slope,
                   activation=ms.nn.ELU)
    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=lr, weight_decay=weight_decay)
    loss = LossNet(net)
    train_net = nn.TrainOneStepCell(loss, optimizer)
    total = 0.
    warm_up = 3
    test_acc = 0.0

    for e in range(epochs):
        beg = time.time()
        train_net.set_train()
        loss_val = train_net(node_feat, node_label, train_mask, *g.get_graph())
        end = time.time()
        dur = end - beg
        if e >= warm_up:
            total = total + dur

        net.set_train(False)
        out = net(node_feat, *g.get_graph()).asnumpy()
        test_mask = ds.test_mask
        labels = ds.node_label
        predict = np.argmax(out[test_mask], axis=1)
        label = labels[test_mask]
        count = np.equal(predict, label)
        epoch_acc = np.sum(count) / label.shape[0]
        if test_acc < epoch_acc:
            test_acc = epoch_acc

        print('epoch:', e, ' test_acc:', test_acc, ' loss:', loss_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GATv2 for whole-graph classification')
    parser.add_argument("--data_path", type=str, default='/home/dataset/', help="path to dataset")
    parser.add_argument("--device", type=str, default="GPU", help="which device to use")
    parser.add_argument("--device_id", type=int, default=0, help="which device id to use")
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate (default: 0.01)')
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
    parser.add_argument('--num_hidden', type=int, default=8, help='num_hidden')
    parser.add_argument('--num_heads', type=int, default=8, help='num_heads')
    parser.add_argument('--num_out_heads', type=int, default=1, help='num_out_heads')
    parser.add_argument('--in_drop', type=float, default=0.6, help='in_drop')
    parser.add_argument('--attn_drop', type=float, default=0.6, help='attn_drop')
    parser.add_argument('--negative_slope', type=float, default=0.2, help='negative_slope')
    args = parser.parse_args()
    print(args)
    main()
