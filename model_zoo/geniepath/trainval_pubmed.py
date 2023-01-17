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
"""train pubmed"""
import argparse
import time
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.context as context

from mindspore_gl.nn import GNNCell
from mindspore_gl import Graph, GraphField
from mindspore_gl.dataset import CoraV2
from src.geniepath import GeniePath, GeniePathLazy

class LossNet(GNNCell):
    """ LossNet definition """
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')

    def construct(self, x, train_mask, target, g: Graph):
        predict = self.net(x, g)
        target = ops.Squeeze()(target)
        loss = self.loss_fn(predict, target)
        loss = loss * train_mask
        return ms.ops.ReduceSum()(loss)

def main(arguments):
    if arguments.fuse and arguments.device == "GPU":
        context.set_context(device_target=arguments.device, save_graphs=True, save_graphs_path="./computational_graph/",
                            mode=context.GRAPH_MODE, enable_graph_kernel=True, device_id=arguments.device_id)
    else:
        context.set_context(device_target=arguments.device, device_id=arguments.device_id)

    dataset = CoraV2(arguments.data_path, 'pubmed')
    train_mask = dataset.train_mask
    test_mask = dataset.test_mask
    val_mask = dataset.val_mask
    train_node_count, test_node_count, val_node_count = int(np.sum(train_mask)), int(np.sum(test_mask)), int(
        np.sum(val_mask))
    adj_coo = dataset[0].adj_coo
    graph = GraphField(
        ms.Tensor(adj_coo[0], dtype=ms.int32),
        ms.Tensor(adj_coo[1], dtype=ms.int32),
        dataset.node_count - 1,
        dataset.edge_count
    )
    node_feat_tensor = ms.Tensor(dataset.node_feat, dtype=ms.float32)
    label_tensor = ms.Tensor(dataset.node_label, dtype=ms.int32)
    train_mask_tensor = ms.Tensor(train_mask, ms.int32)

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
    for e in range(arguments.epochs):
        beg = time.time()
        train_net.set_train(True)
        train_loss = train_net(node_feat_tensor, train_mask_tensor, label_tensor, *graph.get_graph()) / train_node_count
        end = time.time()

        net.set_train(False)
        out = net(node_feat_tensor, *graph.get_graph()).asnumpy()
        predict = np.argmax(out, axis=1)
        train_count = np.sum(np.equal(predict, dataset.node_label) * train_mask)
        val_count = np.sum(np.equal(predict, dataset.node_label) * val_mask)
        test_count = np.sum(np.equal(predict, dataset.node_label) * test_mask)
        print('Epoch {} time:{:.4f} Train loss {:.5f} Train acc: {:.4f} Val acc: {:.4f} Test acc: {:.4f}'
              .format(e, end - beg, float(train_loss.asnumpy()), train_count / train_node_count,
                      val_count / val_node_count, test_count / test_node_count))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GeniePath')
    parser.add_argument("--data_path", type=str, help="path to dataset")
    parser.add_argument("--device", type=str, default="GPU", help="which device to use")
    parser.add_argument("--device_id", type=int, default=0, help="which device id to use")
    parser.add_argument("--hidden_dim", type=int, default=16, help="hidden layer dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="number of GeniePath layers")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.0004, help="learning rate")
    parser.add_argument("--num_attn_head", type=int, default=1, help="number of attention head in GAT function")
    parser.add_argument("--lazy", type=bool, default=True, help="variant GeniePath-Lazy")
    parser.add_argument("--residual", type=bool, default=False, help="use residual for GAT")
    parser.add_argument('--fuse', type=bool, default=True, help="whether use graph mode")

    args = parser.parse_args()
    print(args)
    main(args)
