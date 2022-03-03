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
""" test geniepath """
import numpy as np
import pytest
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.context as context

from mindspore_gl.nn import GNNCell
from mindspore_gl.nn.conv import GATConv
from mindspore_gl import Graph, GraphField
from mindspore_gl.dataset import PubMed

data_path = "/home/workspace/mindspore_dataset/GNN_Dataset/"


class GeniePathLazy(GNNCell):
    """Geniepath lazy"""

    def __init__(self, input_dim, output_dim, hidden_dim=16, num_layers=2, num_attn_head=1):
        super(GeniePathLazy, self).__init__()
        self.hidden_dim = hidden_dim
        self.dense1 = nn.Dense(input_dim, hidden_dim)
        self.dense2 = nn.Dense(hidden_dim, output_dim)
        self.breaths = nn.CellList()
        self.depths = nn.CellList()
        for _ in range(num_layers):
            self.breaths.append(GATConv(hidden_dim, hidden_dim, num_attn_head=num_attn_head))
            self.depths.append(nn.LSTM(hidden_dim * 2, hidden_dim))
        self.tanh = nn.Tanh()
        self.expand = ops.ExpandDims()
        self.cat = ops.Concat(-1)

    def construct(self, x, g: Graph):
        """genie path forward"""
        h = ops.Zeros()((1, ops.Shape()(x)[0], self.hidden_dim), ms.float32)
        c = ops.Zeros()((1, ops.Shape()(x)[0], self.hidden_dim), ms.float32)
        x = self.dense1(x)
        h_tmps = []
        for breath in self.breaths:
            h_tmp = breath(x, g)
            h_tmp = self.tanh(h_tmp)
            h_tmp = self.expand(h_tmp, 0)
            h_tmps.append(h_tmp)
        x = self.expand(x, 0)
        for h_tmp, depth in zip(h_tmps, self.depths):
            in_cat = self.cat((h_tmp, x))
            x, (h, c) = depth(in_cat, (h, c))
        x = x[0]
        x = self.dense2(x)
        return x


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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_geniepath():
    """test geniepath"""
    context.set_context(device_target="GPU", mode=context.GRAPH_MODE, enable_graph_kernel=True)
    epochs = 500
    lr = 0.0004
    num_layers = 2
    hidden_dim = 16
    num_attn_head = 1

    # dataloader
    dataset = PubMed(data_path)

    train_mask = dataset.train_mask
    test_mask = dataset.test_mask
    test_node_count = int(np.sum(test_mask))
    adj_coo = dataset[0].adj_coo
    graph_field = GraphField(
        ms.Tensor(adj_coo[0], dtype=ms.int32),
        ms.Tensor(adj_coo[1], dtype=ms.int32),
        dataset.node_count,
        dataset.edge_count
    )
    node_feat_tensor = ms.Tensor(dataset.node_feat, dtype=ms.float32)
    label_tensor = ms.Tensor(dataset.node_label, dtype=ms.int32)
    train_mask_tensor = ms.Tensor(train_mask, ms.int32)

    net = GeniePathLazy(input_dim=dataset.num_features,
                        output_dim=dataset.num_classes,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        num_attn_head=num_attn_head)

    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=lr)
    loss = LossNet(net)
    train_net = nn.TrainOneStepCell(loss, optimizer)
    for _ in range(epochs):
        train_net.set_train(True)
        train_net(node_feat_tensor, train_mask_tensor, label_tensor, *graph_field.get_graph())

    net.set_train(False)
    out = net(node_feat_tensor, *graph_field.get_graph()).asnumpy()
    predict = np.argmax(out, axis=1)
    test_acc = np.sum(np.equal(predict, dataset.node_label) * test_mask) / test_node_count
    assert test_acc > 0.6
