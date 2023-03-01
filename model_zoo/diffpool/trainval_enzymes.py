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
"""train and eval"""
import argparse
import time
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore.context as context

from mindspore_gl.dataset import Enzymes
from mindspore_gl.nn.gnn_cell import GNNCell
from mindspore_gl.dataloader import RandomBatchSampler
from mindspore_gl import BatchedGraphField, BatchedGraph

from src.utils import TrainOneStepCellWithGradClipping
from src.diffpool import DiffPool
from src.dataset import MultiHomoGraphDataset


class LossNet(GNNCell):
    """ LossNet definition """
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = ms.nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')

    def construct(self, x, label, g: BatchedGraph):
        pred = self.net(x, g)
        return self.net.loss(pred, label, g)


def eval_acc(dataloader, net, np_graph_mask, val_length, batch_size):
    count = 0
    for data in dataloader:
        label, node_feat, row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask = data
        batch_homo = BatchedGraphField(row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask)
        output = net(node_feat, *batch_homo.get_batched_graph()).asnumpy()
        predict = np.argmax(output, axis=1)
        count += np.sum(np.equal(predict, label) * np_graph_mask)
    acc = count / val_length / batch_size
    return acc


def main(arguments):
    if arguments.fuse and arguments.device == "GPU":
        context.set_context(device_target=arguments.device, save_graphs=True, save_graphs_path="./computational_graph/",
                            mode=context.GRAPH_MODE, enable_graph_kernel=True, device_id=arguments.device_id)
    else:
        context.set_context(device_target=arguments.device, device_id=arguments.device_id, mode=context.GRAPH_MODE)
    node_size, edge_size = 1200, 5000
    hidden_dim, embedding_dim = 64, 64
    dataset = Enzymes(arguments.data_path)
    train_batch_sampler = RandomBatchSampler(dataset.train_graphs, batch_size=arguments.batch_size)
    val_batch_sampler = RandomBatchSampler(dataset.val_graphs, batch_size=arguments.batch_size)
    test_batch_sampler = RandomBatchSampler(dataset.test_graphs, batch_size=arguments.batch_size)
    train_length = len(list(train_batch_sampler))
    val_length = len(list(val_batch_sampler))
    test_length = len(list(test_batch_sampler))

    train_graph_dataset = MultiHomoGraphDataset(dataset, arguments.batch_size, node_size=node_size,
                                                edge_size=edge_size, length=train_length)
    val_graph_dataset = MultiHomoGraphDataset(dataset, arguments.batch_size, node_size=node_size,
                                              edge_size=edge_size, length=val_length)
    test_graph_dataset = MultiHomoGraphDataset(dataset, arguments.batch_size, node_size=node_size,
                                               edge_size=edge_size, length=test_length)
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
    net = DiffPool(
        input_dim=dataset.node_feat_size,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        label_dim=dataset.label_dim,
        activation=nn.ReLU(),
        n_layers=arguments.gc_per_block,
        n_pooling=arguments.num_pool,
        linkpred=arguments.linkpred,
        batch_size=arguments.batch_size,
        aggregator_type='mean',
        assign_dim=int(dataset.max_num_node * arguments.pool_ratio),
        pool_ratio=arguments.pool_ratio
    )
    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=arguments.lr)
    loss = LossNet(net)
    train_net = TrainOneStepCellWithGradClipping(loss, optimizer, arguments.clip)
    early_stopping_logger = {"round": -1, "best_val_acc": -1}
    np_graph_mask = [1] * arguments.batch_size
    np_graph_mask.append(0)
    for epoch in range(arguments.epochs):
        start_time = time.time()
        train_net.set_train(True)
        train_loss, total_iter = 0, 0
        for data in train_dataloader:
            label, node_feat, row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask = data
            batch_homo = BatchedGraphField(row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask)
            train_loss += train_net(node_feat, label, *batch_homo.get_batched_graph())
            total_iter += 1
        train_loss /= total_iter
        end_time = time.time()

        train_net.set_train(False)
        train_acc = eval_acc(train_dataloader, net, np_graph_mask, train_length, arguments.batch_size)
        val_acc = eval_acc(val_dataloader, net, np_graph_mask, val_length, arguments.batch_size)
        print('Epoch {}, Time {:.3f} s, Train loss {}, Train acc {:.3f}, '
              'Val acc {:.3f}'.format(epoch, end_time - start_time, train_loss, train_acc, val_acc))

        # early stopper
        if val_acc > early_stopping_logger['best_val_acc']:
            early_stopping_logger['best_val_acc'] = val_acc
            early_stopping_logger['round'] = 0
        else:
            early_stopping_logger['round'] += 1
            print("Early stop: {}/{}, best_acc: {:.3f}".format(early_stopping_logger['round'], arguments.patience,
                                                               early_stopping_logger['best_val_acc']))
            if early_stopping_logger['round'] == arguments.patience:
                break

    test_acc = eval_acc(test_dataloader, net, np_graph_mask, test_length, arguments.batch_size)
    print("Test acc: {:.3f}".format(test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DiffPool')
    parser.add_argument('--data_path', dest='data_path', help='Input Dataset path')
    parser.add_argument("--device", type=str, default="GPU", help="which device to use")
    parser.add_argument("--device_id", type=int, default=0, help="which device id to use")
    parser.add_argument('--pool_ratio', dest='pool_ratio', default=0.10, type=float, help='ratio of pooling')
    parser.add_argument('--num_pool', dest='num_pool', default=1, type=int, help='numbers of pooling layer')
    parser.add_argument('--link_pred', dest='linkpred', default=True, help='switch of link prediction object')
    parser.add_argument('--lr', dest='lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--clip', dest='clip', default=2.0, type=float, help='gradient clipping')
    parser.add_argument('--batch-size', dest='batch_size', default=20, type=int, help='size of batch')
    parser.add_argument('--epochs', dest='epochs', default=4000, type=int, help='numbers of training epoch')
    parser.add_argument('--gc-per-block', dest='gc_per_block', default=3, type=int,
                        help='number of graph conv layer per block')
    parser.add_argument('--patience', type=int, default=250, help="patience to early stop")
    parser.add_argument('--profile', type=bool, default=False, help="feature dimension")
    parser.add_argument('--fuse', type=bool, default=False, help="feature dimension")
    args = parser.parse_args()
    main(args)
