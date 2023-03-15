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
"""GAT model implemented using mindspore-gl"""
import time
import argparse
import os
import sys

import numpy as np
import mindspore as ms
from mindspore.profiler import Profiler
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.context as context
from mindspore.train.callback import TimeMonitor, LossMonitor

from mindspore_gl.nn import GNNCell
from mindspore_gl import Graph

from gnngraph_dataset import GraphDataset

sys.path.append(os.path.join(os.getcwd(), "..", "model_zoo"))
# pylint: disable=C0413
from gat import GatNet


class LossNet(GNNCell):
    """ LossNet definition """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')

    def construct(self, x, target, train_mask, g: Graph):
        """GatNet with loss function"""
        predict = self.net(x, g)
        target = ops.Squeeze()(target)
        loss = self.loss_fn(predict, target)
        loss = loss * train_mask
        return ms.ops.ReduceSum()(loss) / ms.ops.ReduceSum()(train_mask)

class DataNet(ms.nn.Cell):
    """DataNet"""

    def __init__(self, ds, net):
        super().__init__()
        self.x = ds.x
        self.in_deg = ds.in_deg
        self.out_deg = ds.out_deg
        self.train_mask = ms.Tensor(ds.train_mask, ms.float32)
        self.y = ds.y
        self.src_idx = ds.g.src_idx
        self.dst_idx = ds.g.dst_idx
        self.n_nodes = ds.g.n_nodes
        self.n_edges = ds.g.n_edges
        self.net = net

    def construct(self):
        return self.net(self.x, self.y, self.train_mask, self.src_idx, self.dst_idx, self.n_nodes, self.n_edges)

def main(train_args):
    """train procedure"""
    if train_args.fuse:
        context.set_context(device_target=train_args.device, save_graphs=False,
                            save_graphs_path="./computational_graph/",
                            mode=context.GRAPH_MODE, enable_graph_kernel=True,
                            graph_kernel_flags="--enable_expand_ops=Gather --enable_cluster_ops=TensorScatterAdd,"
                                               "UnsortedSegmentSum,GatherNd --enable_recompute_fusion=false "
                                               "--enable_parallel_fusion=true ")
    else:
        context.set_context(device_target=train_args.device, mode=context.PYNATIVE_MODE)

    # dataloader
    ds = GraphDataset(train_args.data_path)
    feature_size = ds.x.shape[1]
    if train_args.profile:
        ms_profiler = Profiler(subgraph="ALL", is_detail=True, is_show_op_path=False, output_path="./prof_result")
    TimeMonitor()
    LossMonitor()

    # model
    net = GatNet(num_layers=train_args.num_layers,
                 data_feat_size=feature_size,
                 hidden_dim_size=train_args.num_hidden,
                 n_classes=ds.n_classes,
                 heads=[train_args.num_heads for _ in range(train_args.num_layers)] + [train_args.num_out_heads],
                 input_drop_out_rate=train_args.in_drop,
                 attn_drop_out_rate=train_args.attn_drop,
                 leaky_relu_slope=train_args.negative_slope,
                 activation=ms.nn.ELU,
                 add_norm=True)
    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=train_args.lr, weight_decay=train_args.weight_decay)
    loss = LossNet(net)
    total = 0.
    warm_up = 3

    grad_fn = ops.value_and_grad(loss, None, optimizer.parameters, has_aux=False)

    @ms.jit
    def train_one_step(x, y, train_mask, src_idx, dst_idx, n_nodes, n_edges):
        loss, grads = grad_fn(x, y, train_mask, src_idx, dst_idx, n_nodes, n_edges)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    train_net = DataNet(ds, train_one_step)
    for e in range(train_args.epochs):
        beg = time.time()
        net.set_train()
        train_loss = train_net()
        end = time.time()
        dur = end - beg
        if e >= warm_up:
            total = total + dur

        test_mask = ds.test_mask
        if test_mask is not None:
            net.set_train(False)
            out = net(ds.x, ds.g.src_idx, ds.g.dst_idx, ds.g.n_nodes, ds.g.n_edges).asnumpy()
            labels = ds.y.asnumpy()
            predict = np.argmax(out[test_mask], axis=1)
            label = labels[test_mask]
            count = np.equal(predict, label)
            print('Epoch time:{} ms Train loss {} Test acc:{}'.format(dur * 1000, train_loss,
                                                                      np.sum(count) / label.shape[0]))
    print("Model:{} Dataset:{} Avg epoch time:{}"
          .format("GAT", train_args.data_path, total * 1000 / (train_args.epochs - warm_up)))
    if train_args.profile:
        ms_profiler.analyse()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--data-path", type=str, default='/home/workspace/cora_v2_with_mask.npz',
                        help="path to dataloader")
    parser.add_argument("--device", type=str, default="GPU", help="which device to use")
    parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8, help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1, help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1, help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8, help="number of hidden units")
    parser.add_argument("--in-drop", type=float, default=.6, help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6, help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--negative-slope", type=float, default=.2, help="the negative slope of leakyRelu")
    parser.add_argument('--profile', type=bool, default=False, help="feature dimension")
    parser.add_argument('--fuse', type=bool, default=False, help="feature dimension")
    args = parser.parse_args()
    print(args)
    main(args)
