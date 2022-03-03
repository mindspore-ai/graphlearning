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
"""Train ppi"""
import argparse
import time
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.context as context

from mindspore_gl.nn import GNNCell
from mindspore_gl import BatchedGraph, BatchedGraphField
from mindspore_gl.dataset import PPI
from mindspore_gl.dataloader import RandomBatchSampler, DataLoader
from sklearn.metrics import f1_score

from src.geniepath import GeniePath, GeniePathLazy


class LossNet(GNNCell):
    """ LossNet definition """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = nn.loss.BCEWithLogitsLoss(reduction='none')

    def construct(self, x, target, g: BatchedGraph):
        """construct function"""
        predict = self.net(x, g)
        loss = self.loss_fn(predict, target)
        loss = loss * ops.Reshape()(g.node_mask(), (-1, 1))
        return ms.ops.ReduceSum()(loss)


def evaluate(net, dataloader, constant_graph_mask, batch_size):
    """evaluate"""
    train_pred, train_label = [], []
    for data in dataloader:
        batch_graph, label, node_feat = data
        node_feat = ms.Tensor.from_numpy(node_feat)
        batch_homo = BatchedGraphField(
            ms.Tensor.from_numpy(batch_graph.adj_coo[0]),
            ms.Tensor.from_numpy(batch_graph.adj_coo[1]),
            ms.Tensor(batch_graph.node_count, ms.int32),
            ms.Tensor(batch_graph.edge_count, ms.int32),
            ms.Tensor.from_numpy(batch_graph.batch_meta.node_map_idx),
            ms.Tensor.from_numpy(batch_graph.batch_meta.edge_map_idx),
            constant_graph_mask
        )
        logits = net(node_feat, *batch_homo.get_batched_graph()).asnumpy()
        predict = np.where(logits >= 0., 1, 0)
        ori_node_count = int(np.sum(batch_graph.batch_meta.node_map_idx < batch_size))
        train_label.extend(label.tolist()[:ori_node_count])
        train_pred.extend(predict.tolist()[:ori_node_count])
    train_micro_f1 = f1_score(train_label, train_pred, average='micro')
    return train_micro_f1


def main(arguments):
    if arguments.fuse:
        context.set_context(device_target="GPU", save_graphs=True, save_graphs_path="./computational_graph/",
                            mode=context.GRAPH_MODE, enable_graph_kernel=True)
    else:
        context.set_context(device_target="GPU")
    # dataloader
    dataset = PPI(arguments.data_path)
    multi_graph_dataset = MultiHomoGraphDataset(dataset, arguments.batch_size)
    train_batch_sampler = RandomBatchSampler(dataset.train_graphs, batch_size=arguments.batch_size)
    train_dataloader = DataLoader(dataset=multi_graph_dataset, sampler=train_batch_sampler, num_workers=4,
                                  persistent_workers=True, prefetch_factor=4)

    test_batch_sampler = RandomBatchSampler(dataset.test_graphs, batch_size=arguments.batch_size)
    test_dataloader = DataLoader(dataset=multi_graph_dataset, sampler=test_batch_sampler, num_workers=0,
                                 persistent_workers=False)

    val_batch_sampler = RandomBatchSampler(dataset.val_graphs, batch_size=arguments.batch_size)
    val_dataloader = DataLoader(dataset=multi_graph_dataset, sampler=val_batch_sampler, num_workers=0,
                                persistent_workers=False)

    if arguments.lazy:
        net = GeniePathLazy(input_dim=dataset.num_features,
                            output_dim=dataset.num_classes,
                            hidden_dim=arguments.hidden_dim,
                            num_layers=arguments.num_layers,
                            num_attn_head=arguments.num_attn_head)
    else:
        net = GeniePath(input_dim=dataset.num_features,
                        output_dim=dataset.num_classes,
                        hidden_dim=arguments.hidden_dim,
                        num_layers=arguments.num_layers,
                        num_attn_head=arguments.num_attn_head)

    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=arguments.lr)
    loss = LossNet(net)
    train_net = nn.TrainOneStepCell(loss, optimizer)

    np_graph_mask = [1] * (arguments.batch_size + 1)
    np_graph_mask[-1] = 0
    constant_graph_mask = ms.Tensor(np_graph_mask, dtype=ms.int32)
    for e in range(arguments.epochs + 1):
        train_net.set_train(True)
        beg = time.time()
        train_loss = 0
        for data in train_dataloader:
            batch_graph, label, node_feat = data
            node_feat = ms.Tensor.from_numpy(node_feat)
            label = ms.Tensor.from_numpy(label)
            batch_homo = BatchedGraphField(
                ms.Tensor.from_numpy(batch_graph.adj_coo[0]),
                ms.Tensor.from_numpy(batch_graph.adj_coo[1]),
                ms.Tensor(batch_graph.node_count, ms.int32),
                ms.Tensor(batch_graph.edge_count, ms.int32),
                ms.Tensor.from_numpy(batch_graph.batch_meta.node_map_idx),
                ms.Tensor.from_numpy(batch_graph.batch_meta.edge_map_idx),
                constant_graph_mask
            )
            origin_node_count = np.sum(batch_graph.batch_meta.node_map_idx < arguments.batch_size)
            train_loss += float(
                train_net(node_feat, label, *batch_homo.get_batched_graph()).asnumpy()) / origin_node_count
        train_loss /= len(train_dataloader) / dataset.num_classes
        end = time.time()

        if e % 10 == 0:
            net.set_train(False)
            train_micro_f1 = evaluate(net, train_dataloader, constant_graph_mask, arguments.batch_size)
            val_micro_f1 = evaluate(net, val_dataloader, constant_graph_mask, arguments.batch_size)
            test_micro_f1 = evaluate(net, test_dataloader, constant_graph_mask, arguments.batch_size)

            print(
                'Epoch {} time: {:.4f} Train loss: {:.5f} Train microF1: {:.4f} Val microF1: {:.4f} '
                'Test microF1: {:.4f}'
                .format(e, end - beg, train_loss, train_micro_f1, val_micro_f1, test_micro_f1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GeniePath')
    parser.add_argument("--data_path", type=str, help="path to dataset")
    parser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    parser.add_argument("--hidden_dim", type=int, default=256, help="hidden layer dimension")
    parser.add_argument("--num_layers", type=int, default=3, help="number of GeniePath layers")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.0004, help="learning rate")
    parser.add_argument("--num_attn_head", type=int, default=1, help="number of attention head in GAT function")
    parser.add_argument("--lazy", type=bool, default=True, help="variant GeniePath-Lazy")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size for dataloader")
    parser.add_argument("--residual", type=bool, default=False, help="use residual for GAT")
    parser.add_argument('--fuse', type=bool, default=True, help="whether use graph mode")

    args = parser.parse_args()
    print(args)
    main(args)
