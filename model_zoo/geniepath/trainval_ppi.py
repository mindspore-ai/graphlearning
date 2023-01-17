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
from sklearn.metrics import f1_score

import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.ops as ops
import mindspore.context as context

from mindspore_gl.nn import GNNCell
from mindspore_gl.dataset import PPI
from mindspore_gl.dataloader import RandomBatchSampler
from mindspore_gl import BatchedGraph, BatchedGraphField

from src.geniepath import GeniePath, GeniePathLazy
from src.dataset import MultiHomoGraphDataset

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


def evaluate(net, dataloader, batch_size):
    """evaluate"""
    train_pred, train_label = [], []
    for data in dataloader:
        label, node_feat, row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask = data
        batch_homo = BatchedGraphField(row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask)
        logits = net(node_feat, *batch_homo.get_batched_graph()).asnumpy()
        predict = np.where(logits >= 0., 1, 0)
        ori_node_count = int(np.sum(node_map_idx.asnumpy() < batch_size))
        train_label.extend(label.asnumpy().tolist()[:ori_node_count])
        train_pred.extend(predict.tolist()[:ori_node_count])
    train_micro_f1 = f1_score(train_label, train_pred, average='micro')
    return train_micro_f1

def main(arguments):
    if arguments.fuse and arguments.device == "GPU":
        context.set_context(device_target=arguments.device, save_graphs=True, save_graphs_path="./computational_graph/",
                            mode=context.GRAPH_MODE, enable_graph_kernel=True, device_id=arguments.device_id)
    else:
        context.set_context(device_target=arguments.device, device_id=arguments.device_id)

    dataset = PPI(arguments.data_path)
    train_batch_sampler = RandomBatchSampler(dataset.train_graphs, batch_size=arguments.batch_size)
    test_batch_sampler = RandomBatchSampler(dataset.test_graphs, batch_size=arguments.batch_size)
    val_batch_sampler = RandomBatchSampler(dataset.val_graphs, batch_size=arguments.batch_size)
    train_length = len(list(train_batch_sampler))
    val_length = len(list(val_batch_sampler))
    test_length = len(list(test_batch_sampler))

    train_graph_dataset = MultiHomoGraphDataset(dataset, arguments.batch_size, length=train_length)
    val_graph_dataset = MultiHomoGraphDataset(dataset, arguments.batch_size, length=val_length)
    test_graph_dataset = MultiHomoGraphDataset(dataset, arguments.batch_size, length=test_length)
    train_dataloader = ds.GeneratorDataset(train_graph_dataset, ['batched_label', 'batched_node_feat', 'row', 'col',
                                                                 'node_count', 'edge_count', 'node_map_idx',
                                                                 'edge_map_idx', 'graph_mask'],
                                           sampler=train_batch_sampler, python_multiprocessing=True)
    val_dataloader = ds.GeneratorDataset(val_graph_dataset, ['batched_label', 'batched_node_feat', 'row', 'col',
                                                             'node_count', 'edge_count', 'node_map_idx',
                                                             'edge_map_idx', 'graph_mask'],
                                         sampler=val_batch_sampler, python_multiprocessing=True)
    test_dataloader = ds.GeneratorDataset(test_graph_dataset, ['batched_label', 'batched_node_feat', 'row', 'col',
                                                               'node_count', 'edge_count', 'node_map_idx',
                                                               'edge_map_idx', 'graph_mask'],
                                          sampler=test_batch_sampler, python_multiprocessing=True)

    if arguments.lazy:
        net = GeniePathLazy(input_dim=dataset.node_feat_size,
                            output_dim=dataset.num_classes,
                            hidden_dim=arguments.hidden_dim,
                            num_layers=arguments.num_layers,
                            num_attn_head=arguments.num_attn_head)
    else:
        net = GeniePath(input_dim=dataset.node_feat_size,
                        output_dim=dataset.num_classes,
                        hidden_dim=arguments.hidden_dim,
                        num_layers=arguments.num_layers,
                        num_attn_head=arguments.num_attn_head)

    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=arguments.lr)
    loss = LossNet(net)
    train_net = nn.TrainOneStepCell(loss, optimizer)
    for e in range(arguments.epochs + 1):
        train_net.set_train(True)
        beg = time.time()
        train_loss = 0
        for data in train_dataloader:
            label, node_feat, row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask = data
            batch_homo = BatchedGraphField(row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask)
            origin_node_count = np.sum(node_map_idx.asnumpy() < arguments.batch_size)
            train_loss += float(
                train_net(node_feat, label, *batch_homo.get_batched_graph()).asnumpy()) / origin_node_count
        train_loss /= train_length / dataset.num_classes
        end = time.time()

        if e % 10 == 0:
            net.set_train(False)
            train_micro_f1 = evaluate(net, train_dataloader, arguments.batch_size)
            val_micro_f1 = evaluate(net, val_dataloader, arguments.batch_size)
            test_micro_f1 = evaluate(net, test_dataloader, arguments.batch_size)
            print(
                'Epoch {} time: {:.4f} Train loss: {:.5f} Train microF1: {:.4f} Val microF1: {:.4f} '
                'Test microF1: {:.4f}'
                .format(e, end - beg, train_loss, train_micro_f1, val_micro_f1, test_micro_f1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GeniePath')
    parser.add_argument("--data_path", type=str, help="path to dataset")
    parser.add_argument("--device", type=str, default="GPU", help="which device to use")
    parser.add_argument("--device_id", type=int, default=0, help="which device id to use")
    parser.add_argument("--hidden_dim", type=int, default=256, help="hidden layer dimension")
    parser.add_argument("--num_layers", type=int, default=3, help="number of GeniePath layers")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
    parser.add_argument("--num_attn_head", type=int, default=1, help="number of attention head in GAT function")
    parser.add_argument("--lazy", type=bool, default=False, help="variant GeniePath-Lazy")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for dataloader")
    parser.add_argument("--residual", type=bool, default=False, help="use residual for GAT")
    parser.add_argument('--fuse', type=bool, default=True, help="whether use graph mode")
    args = parser.parse_args()
    print(args)
    main(args)
