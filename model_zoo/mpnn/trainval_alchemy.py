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
import time
import argparse
import numpy as np
import mindspore as ms
from mindspore.profiler import Profiler
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.ops as ops
import mindspore.context as context

from mindspore_gl.nn import GNNCell
from mindspore_gl import BatchedGraph, BatchedGraphField
from mindspore_gl.dataloader import RandomBatchSampler
from mindspore_gl.dataset import Alchemy

from src.mpnn import MPNNPredictor
from src.dataset import MultiHomoGraphDataset


class LossNet(GNNCell):
    """ LossNet definition """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = nn.MSELoss(reduction='none')

    def construct(self, node_feat, edge_feat, target, bg: BatchedGraph):
        predict = self.net(node_feat, edge_feat, bg)
        target = ops.Squeeze()(target)
        loss = self.loss_fn(predict, target)
        loss = loss * ops.Reshape()(bg.graph_mask, (-1, 1))
        return ms.ops.ReduceMean()(loss)

def build_graph(csr, data):
    """ Build the csr or coo graph """
    if csr:
        label, node_feat, edge_feat, indices, indptr, node_count, edge_count, indices_backward,\
        indptr_backward, node_map_idx, edge_map_idx, graph_mask = data
        # Create ms.Tensor
        batch_homo = BatchedGraphField(indices=indices, indptr=indptr, indices_backward=indices_backward,
                                       indptr_backward=indptr_backward, csr=True, n_nodes=int(node_count),
                                       n_edges=int(edge_count), ver_subgraph_idx=node_map_idx,
                                       edge_subgraph_idx=edge_map_idx, graph_mask=graph_mask)
    else:
        label, node_feat, edge_feat, row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask = data
        batch_homo = BatchedGraphField(src_idx=row, dst_idx=col, n_nodes=int(node_count), n_edges=int(edge_count),
                                       ver_subgraph_idx=node_map_idx,
                                       edge_subgraph_idx=edge_map_idx, graph_mask=graph_mask)
    return label, node_feat, edge_feat, batch_homo

def main(arguments):
    if arguments.device == "GPU" and arguments.fuse:
        if arguments.csr:
            context.set_context(device_target="GPU", save_graphs=False, save_graphs_path="./computational_graph/",
                                mode=context.GRAPH_MODE, enable_graph_kernel=True, device_id=arguments.device_id,
                                graph_kernel_flags="--enable_expand_ops=Gather "
                                                   "--enable_cluster_ops=CSRReduceSum,CSRDiv "
                                                   "--enable_recompute_fusion=false "
                                                   "--enable_parallel_fusion=false "
                                                   "--recompute_increment_threshold=40000000 "
                                                   "--recompute_peak_threshold=3000000000 "
                                                   "--enable_csr_fusion=true ")
        else:
            context.set_context(device_target=arguments.device, save_graphs=False,
                                save_graphs_path="./computational_graph/",
                                mode=context.GRAPH_MODE, enable_graph_kernel=True)
    else:
        context.set_context(device_target=arguments.device, mode=context.GRAPH_MODE, device_id=arguments.device_id)
        if arguments.csr:
            raise ValueError("When the csr operator is used, the fusion function must be enabled.")

    if arguments.profile:
        ms_profiler = Profiler(subgraph="ALL", is_detail=True, is_show_op_path=False, output_path="./prof_result")

    csr = arguments.csr
    backward = arguments.backward
    GNNCell.sparse_compute(csr, backward)

    dataset = Alchemy(arguments.data_path, arguments.data_size)
    train_batch_sampler = RandomBatchSampler(dataset.train_graphs, batch_size=arguments.batch_size)
    test_batch_sampler = RandomBatchSampler(dataset.val_graphs, batch_size=arguments.batch_size)
    train_length = len(list(train_batch_sampler))
    test_length = len(list(test_batch_sampler))
    node_size = arguments.batch_size * 40
    edge_size = arguments.batch_size * 1000
    train_graph_dataset = MultiHomoGraphDataset(dataset, arguments.batch_size, node_size=node_size,
                                                edge_size=edge_size, length=train_length, csr=csr)
    test_graph_dataset = MultiHomoGraphDataset(dataset, arguments.batch_size, node_size=node_size,
                                               edge_size=edge_size, length=test_length, csr=csr)
    if csr:
        columns = ['batched_label', 'batched_node_feat', 'batched_edge_feat', 'indices', 'indptr', 'node_count',
                   'edge_count', 'indices_backward', 'indptr_backward', 'node_map_idx', 'edge_map_idx', 'graph_mask']
    else:
        columns = ['batched_label', 'batched_node_feat', 'batched_edge_feat', 'row', 'col', 'node_count', 'edge_count',
                   'node_map_idx', 'edge_map_idx', 'graph_mask']

    train_dataloader = ds.GeneratorDataset(train_graph_dataset, columns,
                                           sampler=train_batch_sampler, python_multiprocessing=True)
    test_dataloader = ds.GeneratorDataset(test_graph_dataset, columns,
                                          sampler=test_batch_sampler, python_multiprocessing=True)

    # Graph Mask
    np_graph_mask = [[1]] * (arguments.batch_size + 1)
    np_graph_mask[-1] = [0]

    net = MPNNPredictor(node_in_feats=dataset.node_feat_size,
                        edge_in_feats=dataset.edge_feat_size,
                        node_out_feats=arguments.node_out_feats,
                        edge_hidden_feats=arguments.edge_hidden_feats,
                        n_tasks=arguments.n_tasks)

    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=arguments.lr, weight_decay=arguments.weight_decay)
    loss = LossNet(net)
    train_net = nn.TrainOneStepCell(loss, optimizer)
    best_mae = 2e9
    early_stopper = 0

    for epoch in range(arguments.epochs):
        start_time = time.time()
        train_net.set_train(True)
        train_loss = 0
        total_iter = 0
        for data in train_dataloader:
            label, node_feat, edge_feat, batch_homo = build_graph(csr, data)
            train_loss += train_net(node_feat, edge_feat, label, *batch_homo.get_batched_graph()).asnumpy()
            total_iter += 1
        train_loss /= total_iter
        end_time = time.time()

        train_net.set_train(False)
        test_iter = 0
        test_mae = 0
        for data in test_dataloader:
            label, node_feat, edge_feat, batch_homo = build_graph(csr, data)
            output = net(node_feat, edge_feat, *batch_homo.get_batched_graph()).asnumpy()
            test_mae += np.sum(np.abs(output - label.asnumpy()) * np_graph_mask) / arguments.batch_size / \
                        arguments.n_tasks
            test_iter += 1
        test_mae /= test_iter
        print('Epoch {}, Time {:.3f} s, Train loss {}, Test mae {:.3f}'.format(epoch, end_time - start_time, train_loss,
                                                                               test_mae))
        # early stop
        if test_mae < best_mae:
            best_mae = test_mae
            early_stopper = 0
        else:
            early_stopper += 1
            print('Early stop: {}/{}'.format(early_stopper, arguments.patience))
            if early_stopper == arguments.patience:
                break

    if arguments.profile:
        ms_profiler.analyse()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MPNN")
    parser.add_argument("--data_path", type=str, help="path to dataset")
    parser.add_argument("--dataset", type=str, default="Alchemy", help="path to dataloader")
    parser.add_argument("--device", type=str, default="GPU", help="which device to use")
    parser.add_argument("--device_id", type=int, default=0, help="which device id to use")
    parser.add_argument("--epochs", type=int, default=250, help="number of training epochs")
    parser.add_argument('--profile', type=bool, default=False, help="feature dimension")
    parser.add_argument('--fuse', type=bool, default=False, help="enable fusion")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size for graphs")
    parser.add_argument('--node_out_feats', type=int, default=64, help="number of node output features")
    parser.add_argument('--edge_hidden_feats', type=int, default=128, help="number of edge hidden features")
    parser.add_argument('--n_tasks', type=int, default=12, help="number of tasks")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--patience", type=int, default=50, help="number of patience to early stop")
    parser.add_argument("--weight-decay", type=float, default=0, help="weight decay")
    parser.add_argument("--data_size", type=int, default=35000, help="select the size of dataset to use")
    parser.add_argument("--csr", type=bool, default=False, help="whether use the csr operator")
    parser.add_argument("--backward", default=True, action='store_false', help="whether use the customization back"
                                                                               "propagation when use the csr operator")
    args = parser.parse_args()
    print(args)
    main(args)
