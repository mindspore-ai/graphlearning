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
import os
import argparse
import time
import numpy as np

import mindspore.nn as nn
import mindspore.context as context
import mindspore as ms
from mindspore.nn import Cell
import mindspore.dataset as ds
from mindspore.profiler import Profiler
from mindspore.communication import init, get_rank, get_group_size

from mindspore_gl.dataset.reddit import Reddit
from mindspore_gl.dataloader.samplers import RandomBatchSampler, DistributeRandomBatchSampler

from src.gcn import GCNNet
from src.dataset import GCNDataset


device_target = str(os.getenv('DEVICE_TARGET'))
ms.set_context(mode=ms.GRAPH_MODE, device_target=device_target, save_graphs=True, save_graphs_path="./saved_ir/")
if device_target == 'Ascend':
    single_size = True
    device_id = int(os.getenv('DEVICE_ID'))
    ms.set_context(device_id=device_id)
    init()
else:
    init("nccl")
    single_size = False


class LossNet(Cell):
    """ LossNet definition """
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')

    def construct(self, seeds_idx, node_feat, labels, in_deg, out_deg, edges, n_nodes, n_edges):
        out = self.net(node_feat, in_deg, out_deg, edges[0], edges[1], n_nodes, n_edges)
        target_out = out[seeds_idx]
        loss = self.loss_fn(target_out, labels)
        return ms.ops.ReduceSum()(loss) / len(labels)


class GetDegree(Cell):
    """get degree on subgraphs"""
    def __init__(self):
        super().__init__()
        self.reshape = ms.ops.Reshape()
        self.scatter_add = ms.ops.TensorScatterAdd()
        self.ones = ms.ops.Ones()

    def construct(self, edges, n_nodes, n_edges):
        """get degree from subgraphs"""
        src_idx = edges[0]
        dst_idx = edges[1]
        scatter_src_idx = self.reshape(src_idx, (n_edges, 1))
        scatter_dst_idx = self.reshape(dst_idx, (n_edges, 1))

        out_deg = self.scatter_add(
            self.ones(n_nodes, ms.int32),
            scatter_src_idx,
            self.ones(n_edges, ms.int32)
        )
        in_deg = self.scatter_add(
            self.ones(n_nodes, ms.int32),
            scatter_dst_idx,
            self.ones(n_edges, ms.int32)
        )
        return in_deg, out_deg


def main():
    if args.fuse and device_target == "GPU":
        ms.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True,
                       graph_kernel_flags="--enable_expand_ops=Gather --enable_cluster_ops=TensorScatterAdd,"
                                          "UnsortedSegmentSum,GatherNd --enable_recompute_fusion=false "
                                          "--enable_parallel_fusion=true ")
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
    ms.common.set_seed(1234)
    rank_id = get_rank()
    world_size = get_group_size()
    graph_dataset = Reddit(args.data_path)
    train_sampler = DistributeRandomBatchSampler(rank_id, world_size, data_source=graph_dataset.train_nodes,
                                                 batch_size=args.batch_size)
    test_sampler = RandomBatchSampler(data_source=graph_dataset.test_nodes, batch_size=args.batch_size)
    train_dataset = GCNDataset(graph_dataset, [25, 10], args.batch_size, len(list(train_sampler)), single_size)
    test_dataset = GCNDataset(graph_dataset, [25, 10], args.batch_size, len(list(test_sampler)), single_size)
    train_dataloader = ds.GeneratorDataset(train_dataset, ['seeds_idx', 'label', 'nid_feat', 'edges'],
                                           sampler=train_sampler, python_multiprocessing=True, num_parallel_workers=1)
    test_dataloader = ds.GeneratorDataset(test_dataset, ['seeds_idx', 'label', 'nid_feat', 'edges'],
                                          sampler=test_sampler, python_multiprocessing=True, num_parallel_workers=1)
    model = GCNNet(graph_dataset.node_feat_size, 256, graph_dataset.num_classes, dropout=0.5, activation=ms.nn.ELU)
    optimizer = nn.optim.Adam(model.trainable_params(), learning_rate=args.lr, weight_decay=args.weight_decay)
    loss = LossNet(model)
    train_net = nn.TrainOneStepCell(loss, optimizer)
    degree_ops = GetDegree()
    if args.profile:
        ms_profiler = Profiler(subgraph="ALL", is_detail=True, is_show_op_path=False, output_path="./prof_result")
    for epoch in range(args.epochs):
        start = time.time()
        train_net.set_train(True)
        for iter_num, data in enumerate(train_dataloader):
            seeds_idx, nid_label, nid_feat, edges = data
            n_nodes = nid_feat.shape[0]
            n_edges = edges.shape[1]
            in_deg, out_deg = degree_ops(edges, n_nodes, n_edges)
            train_loss = train_net(seeds_idx, nid_feat, nid_label, in_deg, out_deg, edges, n_nodes, n_edges)
            if iter_num % 10 == 0:
                print(f"rank_id:{rank_id} Iteration/Epoch: {iter_num}:{epoch} train loss: {train_loss}", flush=True)
        end = time.time()
        epoch_time = end - start
        print(f"rank_id:{rank_id} Epoch/Time: {epoch}:{epoch_time}", flush=True)
        train_net.set_train(False)
        total_prediction = 0
        correct_prediction = 0
        for iter_num, data in enumerate(test_dataloader):
            seeds_idx, label, nid_feat, edges = data
            n_nodes = nid_feat.shape[0]
            n_edges = edges.shape[1]
            in_deg, out_deg = degree_ops(edges, n_nodes, n_edges)

            out = model(nid_feat, in_deg, out_deg, edges[0], edges[1], n_nodes, n_edges)
            out = out.asnumpy()
            predict = np.argmax(out[seeds_idx. asnumpy()], axis=1)
            correct_prediction += len(np.nonzero(np.equal(predict, label))[0])
            total_prediction += len(seeds_idx)
        print(f"rank_id:{rank_id} test accuracy : {correct_prediction / total_prediction}", flush=True)
    if args.profile:
        ms_profiler.analyse()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument("--data_path", type=str, help="path to dataloader")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size ")
    parser.add_argument("--epochs", type=int, default=10, help="number of training epochs")
    parser.add_argument("--num-hidden", type=int, default=256, help="number of hidden units")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument('--profile', type=bool, default=False, help="training profiling")
    parser.add_argument('--fuse', type=bool, default=False, help="enable fusion")
    args = parser.parse_args()
    print(args)
    start_run_time = time.time()
    main()
