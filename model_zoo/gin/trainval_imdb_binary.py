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
from mindspore.profiler import Profiler
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.context as context

from mindspore_gl.nn.gnn_cell import GNNCell
from mindspore_gl.dataloader import RandomBatchSampler, DataLoader
from mindspore_gl.dataset import IMDBBinary
from mindspore_gl import BatchedGraph, BatchedGraphField

from src.gin import GinNet
from src.dataset import MultiHomoGraphDataset


class LossNet(GNNCell):
    """ LossNet definition """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')

    def construct(self, node_feat, edge_weight, target, g: BatchedGraph):
        predict = self.net(node_feat, edge_weight, g)
        target = ops.Squeeze()(target)
        loss = self.loss_fn(predict, target)
        ##############################
        # Mask Loss
        ##############################
        return ms.ops.ReduceSum()(loss * g.graph_mask)


def get_batched_graph_field(batch_graph, constant_graph_mask):
    return BatchedGraphField(
        ms.Tensor.from_numpy(batch_graph.adj_coo[0]),
        ms.Tensor.from_numpy(batch_graph.adj_coo[1]),
        ms.Tensor(batch_graph.node_count, ms.int32),
        ms.Tensor(batch_graph.edge_count, ms.int32),
        ms.Tensor.from_numpy(batch_graph.batch_meta.node_map_idx),
        ms.Tensor.from_numpy(batch_graph.batch_meta.edge_map_idx),
        constant_graph_mask
    )


def main(arguments):
    if arguments.fuse:
        context.set_context(device_target="GPU", save_graphs=True, save_graphs_path="./computational_graph/",
                            mode=context.GRAPH_MODE, enable_graph_kernel=True)
    else:
        context.set_context(device_target="GPU")

    if arguments.profile:
        ms_profiler = Profiler(subgraph="ALL", is_detail=True, is_show_op_path=False, output_path="./prof_result")

    dataset = IMDBBinary(arguments.data_path)
    train_batch_sampler = RandomBatchSampler(dataset.train_graphs, batch_size=arguments.batch_size)
    multi_graph_dataset = MultiHomoGraphDataset(dataset, arguments.batch_size)
    train_dataloader = DataLoader(dataset=multi_graph_dataset, sampler=train_batch_sampler, num_workers=4,
                                  persistent_workers=True, prefetch_factor=4)

    test_batch_sampler = RandomBatchSampler(dataset.val_graphs, batch_size=arguments.batch_size)
    test_dataloader = DataLoader(dataset=multi_graph_dataset, sampler=test_batch_sampler, num_workers=0,
                                 persistent_workers=False)

    np_graph_mask = [1] * (arguments.batch_size + 1)
    np_graph_mask[-1] = 0
    constant_graph_mask = ms.Tensor(np_graph_mask, dtype=ms.int32)

    net = GinNet(num_layers=arguments.num_layers,
                 num_mlp_layers=arguments.num_mlp_layers,
                 input_dim=dataset.num_features,
                 hidden_dim=arguments.hidden_dim,
                 output_dim=dataset.num_classes,
                 final_dropout=arguments.final_dropout,
                 learn_eps=arguments.learn_eps,
                 graph_pooling_type=arguments.graph_pooling_type,
                 neighbor_pooling_type=arguments.neighbor_pooling_type)

    learning_rates = nn.piecewise_constant_lr(
        [50, 100, 150, 200, 250, 300, 350], [0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003125, 0.00015625])
    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=learning_rates)
    loss = LossNet(net)
    train_net = nn.TrainOneStepCell(loss, optimizer)

    for epoch in range(arguments.epochs):
        start_time = time.time()
        train_net.set_train(True)
        train_loss = 0
        total_iter = 0
        while True:
            for data in train_dataloader:
                batch_graph, label, node_feat, edge_feat = data
                node_feat = ms.Tensor.from_numpy(node_feat)
                edge_feat = ms.Tensor.from_numpy(edge_feat)
                label = ms.Tensor.from_numpy(label)
                batch_homo = get_batched_graph_field(batch_graph, constant_graph_mask)
                train_loss += train_net(node_feat, edge_feat, label, *batch_homo.get_batched_graph())\
                    / arguments.batch_size
                total_iter += 1
                if total_iter == arguments.iters_per_epoch:
                    break
            if total_iter == arguments.iters_per_epoch:
                break
        train_loss /= arguments.iters_per_epoch
        train_net.set_train(False)
        train_count = 0
        for data in train_dataloader:
            batch_graph, label, node_feat, edge_feat = data
            node_feat = ms.Tensor.from_numpy(node_feat)
            edge_feat = ms.Tensor.from_numpy(edge_feat)
            batch_homo = get_batched_graph_field(batch_graph, constant_graph_mask)
            output = net(node_feat, edge_feat, *batch_homo.get_batched_graph()).asnumpy()
            predict = np.argmax(output, axis=1)
            train_count += np.sum(np.equal(predict, label) * np_graph_mask)
        train_acc = train_count / len(train_dataloader) / arguments.batch_size
        end_time = time.time()

        test_count = 0
        for data in test_dataloader:
            batch_graph, label, node_feat, edge_feat = data
            node_feat = ms.Tensor.from_numpy(node_feat)
            edge_feat = ms.Tensor.from_numpy(edge_feat)
            batch_homo = get_batched_graph_field(batch_graph, constant_graph_mask)
            output = net(node_feat, edge_feat, *batch_homo.get_batched_graph()).asnumpy()
            predict = np.argmax(output, axis=1)
            test_count += np.sum(np.equal(predict, label) * np_graph_mask)

        test_acc = test_count / len(test_dataloader) / arguments.batch_size
        print('Epoch {}, Time {:.3f} s, Train loss {}, Train acc {:.3f}, Test acc {:.3f}'.format(epoch,
                                                                                                 end_time - start_time,
                                                                                                 train_loss, train_acc,
                                                                                                 test_acc))
    print(f"check time per epoch {(time.time() - start_time) / arguments.epochs}")
    if arguments.profile:
        ms_profiler.analyse()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Graph convolutional neural net for whole-graph classification')
    parser.add_argument("--data_path", type=str, help="path to dataset")
    parser.add_argument("--dataset", type=str, default="IMDBBINARY", help="dataset")
    parser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    parser.add_argument('--epochs', type=int, default=350, help='number of epochs to train (default: 350)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64, help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5, help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "avg"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "avg", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true", default=True,
                        help='Whether to learn the epsilon weighting for the center nodes. '
                             'Does not affect training accuracy though.')
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument('--profile', type=bool, default=False, help="feature dimension")
    parser.add_argument('--fuse', type=bool, default=True, help="feature dimension")
    args = parser.parse_args()
    print(args)
    main(args)
