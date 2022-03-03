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
import numpy as np
import mindspore.nn as nn
import mindspore.context as context
import mindspore as ms
from mindspore.nn import Cell

from mindspore_gl.dataset import Reddit
from mindspore_gl.dataloader import RandomBatchSampler
from mindspore_gl.dataloader import DataLoader

from src.graphsage import SAGENet
from src.dataset import GraphSAGEDataset


class LossNet(Cell):
    """ LossNet definition """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')

    def construct(self, seeds_idx, node_feat, labels, layered_edges_0, layered_edges_1):
        out = self.net(node_feat, layered_edges_0, layered_edges_1)
        target_out = out[seeds_idx]
        loss = self.loss_fn(target_out, labels)
        return ms.ops.ReduceSum()(loss) / len(labels)


def main(arguments):
    if arguments.fuse:
        context.set_context(device_target="GPU", save_graphs=True, save_graphs_path="./computational_graph/",
                            mode=context.GRAPH_MODE, enable_graph_kernel=True)
    else:
        context.set_context(device_target="GPU", mode=context.GRAPH_MODE, save_graphs=True,
                            save_graphs_path="./saved_ir/")

    ##########################
    # SET UP DATA PIPELINE
    ##########################
    graph_dataset = Reddit(arguments.data_path)
    train_sampler = RandomBatchSampler(data_source=graph_dataset.train_nodes, batch_size=arguments.batch_size)
    test_sampler = RandomBatchSampler(data_source=graph_dataset.test_nodes, batch_size=arguments.batch_size)
    dataset = GraphSAGEDataset(graph_dataset, [25, 10], arguments.batch_size)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, num_workers=1)

    test_dataloader = DataLoader(dataset, sampler=test_sampler, num_workers=0)
    #######################################
    # DEFINE MODEL
    #######################################
    model = SAGENet(graph_dataset.num_features, 256, graph_dataset.num_classes)
    optimizer = nn.optim.Adam(model.trainable_params(), learning_rate=arguments.lr, weight_decay=arguments.weight_decay)
    loss = LossNet(model)
    train_net = nn.TrainOneStepCell(loss, optimizer)

    ##################################
    # TRAIN MODEL
    ##################################
    train_net.set_train(True)
    for epoch in range(arguments.epochs):
        for iter_num, data in enumerate(train_dataloader):
            seeds_idx, label, nid_feat, layered_edges_0, layered_edges_1 = data
            ###########################
            # Transform to  Tensor
            ###########################
            seeds_idx = ms.Tensor.from_numpy(seeds_idx)
            nid_feat = ms.Tensor.from_numpy(nid_feat)
            nid_label = ms.Tensor.from_numpy(label)
            layered_edges_0 = ms.Tensor.from_numpy(layered_edges_0)
            layered_edges_1 = ms.Tensor.from_numpy(layered_edges_1)
            train_loss = train_net(seeds_idx, nid_feat, nid_label, layered_edges_0, layered_edges_1)
            if iter_num % 10 == 0:
                print(f"Iteration/Epoch: {iter_num}:{epoch} train loss: {train_loss}")

    ##################################
    # TEST MODEL
    ##################################
    total_prediction = 0
    correct_prediction = 0
    train_net.set_train(False)
    for iter_num, data in enumerate(test_dataloader):
        seeds_idx, label, nid_feat, layered_edges_0, layered_edges_1 = data
        ###########################
        # Transform to  Tensor
        ###########################
        nid_feat = ms.Tensor.from_numpy(nid_feat)
        layered_edges_0 = ms.Tensor.from_numpy(layered_edges_0)
        layered_edges_1 = ms.Tensor.from_numpy(layered_edges_1)

        out = model(nid_feat, layered_edges_0, layered_edges_1)
        out = out.asnumpy()
        predict = np.argmax(out[seeds_idx], axis=1)
        correct_prediction += len(np.nonzero(np.equal(predict, label))[0])
        total_prediction += len(seeds_idx)
    print(f"test accuracy : {correct_prediction / total_prediction}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--data-path", type=str, help="path to dataloader")
    parser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size ")
    parser.add_argument("--dropout", type=float, default=0.5, help="drop out keep rate")
    parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--num-layers", type=int, default=1, help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=16, help="number of hidden units")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument('--profile', type=bool, default=False, help="feature dimension")
    parser.add_argument('--fuse', type=bool, default=False, help="enable fusion")
    args = parser.parse_args()
    print(args)
    main(args)
