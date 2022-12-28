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
import mindspore.nn as nn
import mindspore.context as context

from mindspore_gl import HeterGraphField
from src.han import HAN


class LossNet(ms.nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = ms.nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    def construct(self, h, target, train_idx, src_idx, dst_idx, n_nodes, n_edges):
        pred = self.net(h, src_idx, dst_idx, n_nodes, n_edges)
        loss = self.loss_fn(pred[train_idx], target)
        return loss


def main(arguments):
    context.set_context(device_target="GPU", mode=context.PYNATIVE_MODE)
    if arguments.fuse:
        context.set_context(device_target="GPU", mode=context.GRAPH_MODE, enable_graph_kernel=True)
    context.set_context(device_id=1)
    npz = np.load(arguments.data_path)
    features = ms.Tensor(npz["features"], ms.float32)
    src_idx = [ms.Tensor(npz["pap_sid"], ms.int32), ms.Tensor(npz["plp_sid"], ms.int32)]
    dst_idx = [ms.Tensor(npz["pap_did"], ms.int32), ms.Tensor(npz["plp_did"], ms.int32)]
    num_nodes = int(npz['num_nodes'])
    n_classes = int(npz['n_classes'])
    feat_dim = int(npz['feat_dim'])
    train_idx = ms.Tensor(npz['train_idx'])
    test_idx = ms.Tensor(npz['test_idx'])
    labels = ms.Tensor(npz['labels'])
    train_labels = labels[train_idx]
    n_nodes = [num_nodes, num_nodes]
    n_edges = [ms.ops.Shape()(dst_idx[0])[0], ms.ops.Shape()(dst_idx[1])[0]]
    hgf = HeterGraphField(src_idx, dst_idx, n_nodes, n_edges)
    print(
        "n_classes:{}, num_nodes:{}, train_idx:{} test_idx:{}, labels:{} train_labels:{} features:{}"
        " feature_dim:{}".format(
            n_classes, num_nodes, train_idx.shape, test_idx.shape, labels.shape, train_labels.shape, features.shape,
            feat_dim))
    net = HAN(2, feat_dim, arguments.hidden_size, n_classes, arguments.num_heads, arguments.drop_out)
    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=arguments.lr, weight_decay=arguments.weight_decay)
    loss = LossNet(net)
    train_net = nn.TrainOneStepCell(loss, optimizer)
    warm_up = 3
    total = 0.
    for e in range(arguments.epochs):
        beg = time.time()
        train_net.set_train()
        train_loss = train_net(features, train_labels, train_idx, *hgf.get_heter_graph())

        end = time.time()
        dur = end - beg
        if e >= warm_up:
            total = total + dur

        net.set_train(False)
        out = net(features, *hgf.get_heter_graph())
        test_predict = out[test_idx].asnumpy().argmax(axis=1)
        test_label = labels[test_idx].asnumpy()
        count = np.equal(test_predict, test_label)
        print('Epoch:{} Epoch time:{} ms Train loss {} Test acc:{}'.format(e, dur * 1000, train_loss,
                                                                           np.sum(count) / test_label.shape[0]))
    print("Model:{} Dataset:{} Avg epoch time:{}".format("HAN", arguments.data_path,
                                                         total * 1000 / (arguments.epochs - warm_up)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HAN")
    parser.add_argument('--data_path', type=str, help="Path to data file", required=True)
    parser.add_argument('--lr', type=float, help="Learning rate", default=0.005)
    parser.add_argument('--num_heads', type=list, help="Number of attention heads for each layer", default=[8])
    parser.add_argument('--hidden_size', type=int, help="Number of hidden units", default=8)
    parser.add_argument('--drop_out', type=float, help="Drop out rate", default=0.2)
    parser.add_argument('--weight_decay', type=float, help="Weight decay", default=0.001)
    parser.add_argument('--epochs', type=int, help="Number of training epochs", default=200)
    parser.add_argument('--fuse', type=bool, default=False, help="enable fusion")
    args = parser.parse_args()
    main(args)
