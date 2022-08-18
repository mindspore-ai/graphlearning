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
"""train eval"""
import argparse
import time
import numpy as np
from sklearn.model_selection import train_test_split
import mindspore as ms
import mindspore.nn as nn
import mindspore.context as context
import mindspore.ops as ops
from mindspore.profiler import Profiler
from mindspore_gl.dataset.metr_la import MetrLa
from mindspore_gl.graph import norm
from mindspore_gl import Graph, GraphField
from mindspore_gl.nn.gnn_cell import GNNCell

from src import STGcnNet

class LossNet(GNNCell):
    """ LossNet definition """
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = nn.loss.MSELoss()

    def construct(self, feat, edges, target, g: Graph):
        """STGCN Net with loss function"""
        predict = self.net(feat, edges, g)
        predict = ops.Squeeze()(predict)
        loss = self.loss_fn(predict, target)
        return ms.ops.ReduceMean()(loss)

def main():
    if args.fuse and args.device == "GPU":
        context.set_context(device_target="GPU", save_graphs=True, save_graphs_path="./computational_graph/",
                            mode=context.GRAPH_MODE, enable_graph_kernel=True,
                            graph_kernel_flags="--enable_expand_ops=Gather --enable_cluster_ops=TensorScatterAdd,"
                                               "UnsortedSegmentSum,GatherNd --enable_recompute_fusion=false "
                                               "--enable_parallel_fusion=true ")
    else:
        context.set_context(device_target=args.device, mode=context.GRAPH_MODE, save_graphs=True,
                            save_graphs_path="./saved_ir/")
    epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay
    batch_size = args.batch_size

    metr = MetrLa(args.data_path)
    # out_timestep setting
    # out_timestep = in_timestep - ((kernel_size - 1) * 2 * layer_nums)
    # such as: layer_nums = 2, kernel_size = 3, in_timestep = 12,
    # out_timestep = 4
    features, labels = metr.get_data(args.in_timestep, args.out_timestep)
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True)
    edge_index = metr.edge_index
    edge_attr = metr.edge_attr
    node_num = metr.node_num
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    edge_attr = edge_attr[mask]

    edge_index = ms.Tensor(edge_index, ms.int32)
    edge_attr = ms.Tensor(edge_attr, ms.float32)
    edge_index, edge_weight = norm(edge_index, node_num, edge_attr, args.normalization)
    edge_weight = ms.ops.Reshape()(edge_weight, ms.ops.Shape()(edge_weight) + (1,))
    g = GraphField(edge_index[0], edge_index[1], node_num, len(edge_index[0]))

    net = STGcnNet(num_nodes=node_num,
                   in_channels=args.in_channels,
                   hidden_channels_1st=args.hidden_channels_1st,
                   out_channels_1st=args.out_channels_1st,
                   hidden_channels_2nd=args.hidden_channels_2nd,
                   out_channels_2nd=args.out_channels_2nd,
                   out_channels=args.out_channels,
                   kernel_size=args.kernel_size,
                   k=args.k)

    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=lr, weight_decay=weight_decay)
    loss = LossNet(net)
    train_net = nn.TrainOneStepCell(loss, optimizer)
    if args.profile:
        ms_profiler = Profiler(subgraph="ALL", is_detail=True, is_show_op_path=False, output_path="./prof_result")

    for epoch in range(epochs):
        c = 1
        loss_list = []
        beg = time.time()
        for i in range(0, len(x_train), batch_size):
            train_net.set_train()
            node_feat = x_train[i: i + batch_size]
            node_target = y_train[i: i + batch_size]
            node_feat = np.transpose(node_feat, (0, 3, 1, 2))
            node_target = np.transpose(node_target, (0, 2, 1))
            node_feat = ms.Tensor(node_feat, ms.float32)
            node_target = ms.Tensor(node_target, ms.float32)
            train_loss = train_net(node_feat, edge_weight, node_target, *g.get_graph())
            loss_list.append(train_loss)
            if c % 100 == 0:
                print(f"Iteration/Epoch: {c}:{epoch} loss: {sum(loss_list) / len(loss_list)}")
            c += 1
        end = time.time()
        print('Time', end - beg, 'Epoch loss', sum(loss_list) / len(loss_list))

    net.set_train(False)
    loss_list = []
    for j in range(0, len(x_test), batch_size):
        node_feat = x_test[j: j + batch_size]
        node_target = y_test[j: j + batch_size]
        node_feat = np.transpose(node_feat, (0, 3, 1, 2))
        node_target = np.transpose(node_target, (0, 2, 1))
        node_feat = ms.Tensor(node_feat, ms.float32)
        node_target = ms.Tensor(node_target, ms.float32)
        train_loss = train_net(node_feat, edge_weight, node_target, *g.get_graph())
        loss_list.append(train_loss)
    print("eval MSE:", sum(loss_list) / len(loss_list))
    if args.profile:
        ms_profiler.analyse()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--data-path", type=str, help="path to dataloader")
    parser.add_argument("--gpu", type=int, default=4, help="which gpu to use")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size ")
    parser.add_argument("--epochs", type=int, default=1, help="number of training epochs")
    parser.add_argument("--in-timestep", type=int, default=12, help="length of input timestep")
    parser.add_argument("--out-timestep", type=int, default=4, help="length of output timestep")
    parser.add_argument("--in-channels", type=int, default=2, help="number of input units")
    parser.add_argument("--hidden-channels_1st", type=int, default=64, help="number of hidden units of 1st layer")
    parser.add_argument("--out_channels-1st", type=int, default=32, help="number of output units of 1st layer")
    parser.add_argument("--hidden_channels-2nd", type=int, default=16, help="number of hidden units of 2nd layer")
    parser.add_argument("--out_channels-2nd", type=int, default=8, help="number of output units of 1nd layer")
    parser.add_argument("--out-channels", type=int, default=1, help="number of output units")
    parser.add_argument("--kernel-size", type=int, default=3, help="Convolutional kernel size")
    parser.add_argument("--k", type=int, default=3, help="Chebyshev filter size")
    parser.add_argument("--normalization", type=str, default='sym', help="graph laplace normalization")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument('--profile', type=bool, default=False, help="training profiling")
    parser.add_argument('--fuse', type=bool, default=False, help="enable fusion")
    parser.add_argument("--device", type=str, default="GPU", help="which device to use")
    args = parser.parse_args()
    print(args)
    main()
