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
import mindspore.nn as nn
import mindspore.context as context

from mindspore_gl.dataset import Enzymes
from mindspore_gl.nn.gnn_cell import GNNCell
from mindspore_gl.dataloader import RandomBatchSampler, DataLoader
from mindspore_gl import BatchedGraphField, BatchedGraph

from src.utils import TrainOneStepCellWithGradClipping
from src.diffpool import DiffPool
from src.dataset import MultiHomoGraphDataset

class LossNet(GNNCell):
    """ LossNet definition """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')

    def construct(self, x, label, g: BatchedGraph):
        pred = self.net(x, g)
        return self.net.loss(pred, label, g)

def get_batch_graph_field(batch_graph, node_loop, constant_graph_mask):
    return BatchedGraphField(
        ms.Tensor.from_numpy(np.concatenate((batch_graph.adj_coo[0], node_loop))),
        ms.Tensor.from_numpy(np.concatenate((batch_graph.adj_coo[1], node_loop))),
        ms.Tensor(batch_graph.node_count, ms.int32),
        ms.Tensor(batch_graph.edge_count + batch_graph.node_count, ms.int32),
        ms.Tensor.from_numpy(batch_graph.batch_meta.node_map_idx),
        ms.Tensor.from_numpy(
            np.concatenate((batch_graph.batch_meta.edge_map_idx, batch_graph.batch_meta.node_map_idx))),
        constant_graph_mask
    )

def main(arguments):
    if arguments.fuse:
        context.set_context(device_target="GPU", save_graphs=True, save_graphs_path="./computational_graph/",
                            mode=context.GRAPH_MODE, enable_graph_kernel=True)
    else:
        context.set_context(device_target="GPU")
    node_size, edge_size = 1200, 5000
    hidden_dim, embedding_dim = 64, 64
    dataset = Enzymes(arguments.data_path)
    train_batch_sampler = RandomBatchSampler(dataset.train_graphs, batch_size=arguments.batch_size)
    multi_graph_dataset = MultiHomoGraphDataset(dataset, arguments.batch_size, node_size=node_size, edge_size=edge_size)
    train_dataloader = DataLoader(dataset=multi_graph_dataset, sampler=train_batch_sampler, num_workers=1,
                                  persistent_workers=True)
    val_batch_sampler = RandomBatchSampler(dataset.val_graphs, batch_size=arguments.batch_size)
    val_dataloader = DataLoader(dataset=multi_graph_dataset, sampler=val_batch_sampler, num_workers=0)
    test_batch_sampler = RandomBatchSampler(dataset.test_graphs, batch_size=arguments.batch_size)
    test_dataloader = DataLoader(dataset=multi_graph_dataset, sampler=test_batch_sampler, num_workers=0)

    np_graph_mask = [1] * arguments.batch_size
    np_graph_mask.append(0)
    constant_graph_mask = ms.Tensor(np_graph_mask, dtype=ms.int32)

    net = DiffPool(
        input_dim=dataset.num_features,
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
    node_loop = np.arange(0, node_size, dtype=np.int32)
    for epoch in range(arguments.epochs):
        start_time = time.time()
        train_net.set_train(True)
        train_loss, total_iter = 0, 0
        for data in train_dataloader:
            batch_graph, label, node_feat = data
            node_feat = ms.Tensor.from_numpy(node_feat)
            label = ms.Tensor.from_numpy(label)
            batch_homo = get_batch_graph_field(batch_graph, node_loop, constant_graph_mask)
            train_loss += train_net(node_feat, label, *batch_homo.get_batched_graph())
            total_iter += 1
        train_loss /= total_iter
        end_time = time.time()

        train_net.set_train(False)
        train_count = 0
        for data in train_dataloader:
            batch_graph, label, node_feat = data
            node_feat = ms.Tensor.from_numpy(node_feat)
            batch_homo = get_batch_graph_field(batch_graph, node_loop, constant_graph_mask)
            output = net(node_feat, *batch_homo.get_batched_graph()).asnumpy()
            predict = np.argmax(output, axis=1)
            train_count += np.sum(np.equal(predict, label) * np_graph_mask)
        train_acc = train_count / len(train_dataloader) / arguments.batch_size

        val_count = 0
        for data in val_dataloader:
            batch_graph, label, node_feat = data
            node_feat = ms.Tensor.from_numpy(node_feat)
            batch_homo = get_batch_graph_field(batch_graph, node_loop, constant_graph_mask)
            output = net(node_feat, *batch_homo.get_batched_graph()).asnumpy()
            predict = np.argmax(output, axis=1)
            val_count += np.sum(np.equal(predict, label) * np_graph_mask)
        val_acc = val_count / len(val_dataloader) / arguments.batch_size
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

    test_count = 0
    for data in test_dataloader:
        batch_graph, label, node_feat = data
        node_feat = ms.Tensor(node_feat, ms.float32)
        batch_homo = get_batch_graph_field(batch_graph, node_loop, constant_graph_mask)
        output = net(node_feat, *batch_homo.get_batched_graph()).asnumpy()
        predict = np.argmax(output, axis=1)
        test_count += np.sum(np.equal(predict, label) * np_graph_mask)
    test_acc = test_count / len(test_dataloader) / arguments.batch_size
    print("Test acc: {:.3f}".format(test_acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DiffPool')
    parser.add_argument('--data_path', dest='data_path', help='Input Dataset path')
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
