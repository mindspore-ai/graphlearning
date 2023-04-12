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
import math
import numpy as np
import mindspore.nn as nn
import mindspore.context as context
import mindspore as ms
import mindspore.dataset as ds
from mindspore.nn import Cell
from mindspore.profiler import Profiler

from mindspore_gl.dataset import Reddit
from mindspore_gl.dataloader.samplers import RandomBatchSampler

from src.graphsage import SAGENet
from src.dataset import GraphSAGEDataset


class LossNet(Cell):
    """ LossNet definition """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')

    def construct(self, seeds_idx, node_feat, labels, edges, n_nodes, n_edges):
        out = self.net(node_feat, edges, n_nodes, n_edges)
        target_out = out[seeds_idx]
        loss = self.loss_fn(target_out, labels)
        return ms.ops.ReduceSum()(loss) / len(labels)


def main():
    if args.fuse and args.device == "GPU":
        context.set_context(device_target="GPU", save_graphs=True, save_graphs_path="./computational_graph/",
                            mode=context.GRAPH_MODE, enable_graph_kernel=True, device_id=args.device_id,
                            graph_kernel_flags="--enable_expand_ops=Gather --enable_cluster_ops=TensorScatterAdd,"
                                               "UnsortedSegmentSum,GatherNd --enable_recompute_fusion=false "
                                               "--enable_parallel_fusion=true ")
    else:
        context.set_context(device_target=args.device, mode=context.GRAPH_MODE, save_graphs=True,
                            save_graphs_path="./saved_ir/", device_id=args.device_id)

    graph_dataset = Reddit(args.data_path)
    train_sampler = RandomBatchSampler(data_source=graph_dataset.train_nodes, batch_size=args.batch_size)
    test_sampler = RandomBatchSampler(data_source=graph_dataset.test_nodes, batch_size=args.batch_size)
    train_dataset = GraphSAGEDataset(graph_dataset, [25, 10], args.batch_size, len(list(train_sampler)))
    test_dataset = GraphSAGEDataset(graph_dataset, [25, 10], args.batch_size, len(list(test_sampler)))
    train_dataloader = ds.GeneratorDataset(train_dataset, ['seeds_idx', 'label', 'nid_feat', 'edges'],
                                           sampler=train_sampler, python_multiprocessing=True)
    test_dataloader = ds.GeneratorDataset(test_dataset, ['seeds_idx', 'label', 'nid_feat', 'edges'],
                                          sampler=test_sampler, python_multiprocessing=True)

    appr_dim = math.ceil(graph_dataset.num_classes/8)*8
    model = SAGENet(graph_dataset.node_feat_size, 256, appr_dim, graph_dataset.num_classes)
    optimizer = nn.optim.Adam(model.trainable_params(), learning_rate=args.lr, weight_decay=args.weight_decay)
    loss = LossNet(model)
    train_net = nn.TrainOneStepCell(loss, optimizer)
    if args.profile:
        ms_profiler = Profiler(subgraph="ALL", is_detail=True, is_show_op_path=False, output_path="./prof_result")

    for epoch in range(args.epochs):
        start = time.time()
        train_net.set_train(True)
        for iter_num, data in enumerate(train_dataloader):
            seeds_idx, nid_label, nid_feat, edges = data
            n_nodes = nid_feat.shape[0]
            n_edges = edges.shape[1]
            train_loss = train_net(seeds_idx, nid_feat, nid_label, edges, n_nodes, n_edges)
            if iter_num % 10 == 0:
                print(f"Iteration/Epoch: {iter_num}:{epoch} train loss: {train_loss}")
        end = time.time()
        epoch_time = end - start
        print(f"Epoch/Time: {epoch}:{epoch_time}")

        total_prediction = 0
        correct_prediction = 0
        train_net.set_train(False)
        for iter_num, data in enumerate(test_dataloader):
            seeds_idx, label, nid_feat, edges = data
            n_nodes = nid_feat.shape[0]
            n_edges = edges.shape[1]
            out = model(nid_feat, edges, n_nodes, n_edges)
            out = out.asnumpy()
            predict = np.argmax(out[seeds_idx. asnumpy()], axis=1)
            correct_prediction += len(np.nonzero(np.equal(predict, label))[0])
            total_prediction += len(seeds_idx)
        print(f"test accuracy : {correct_prediction / total_prediction}")
    ms.export(model, nid_feat, edges, n_nodes, n_edges, file_name="sage_model", file_format="MINDIR")

    if args.profile:
        ms_profiler.analyse()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graphsage")
    parser.add_argument("--data-path", type=str, help="path to dataloader")
    parser.add_argument("--device_id", type=int, default=0, help="which device id to use")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size ")
    parser.add_argument("--dropout", type=float, default=0.5, help="drop out rate")
    parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--num-layers", type=int, default=1, help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=16, help="number of hidden units")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument('--profile', type=bool, default=False, help="training profiling")
    parser.add_argument('--fuse', type=bool, default=False, help="enable fusion")
    parser.add_argument("--device", type=str, default="GPU", help="which device to use")
    args = parser.parse_args()
    print(args)
    main()
