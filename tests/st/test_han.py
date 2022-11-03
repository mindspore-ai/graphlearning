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
""" test han """
import time
import pytest
import numpy as np

import mindspore as ms
import mindspore.nn as nn

import mindspore.context as context

from mindspore_gl.nn import GATConv

data_path = "/home/workspace/mindspore_dataset/GNN_Dataset/ACM3025.npz"


class SemanticAttention(ms.nn.Cell):
    """ semantic attention """

    def __init__(self,
                 in_feat_size,
                 hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.proj = ms.nn.SequentialCell(
            ms.nn.Dense(in_feat_size, hidden_size),
            ms.nn.Tanh(),
            ms.nn.Dense(hidden_size, 1, has_bias=False)
        )

    def construct(self, x):
        h = ms.ops.ReduceMean()(self.proj(x), 0)
        beta = ms.ops.Softmax(0)(h)
        beta = ms.ops.BroadcastTo((ms.ops.Shape()(x)[0],) + ms.ops.Shape()(beta))(beta)
        return ms.ops.ReduceSum()(beta * x, 1)


class HANLayer(ms.nn.Cell):
    """ han layer """

    def __init__(self,
                 num_meta_paths,
                 in_feat_size,
                 out_size,
                 num_heads,
                 dropout):
        super(HANLayer, self).__init__()
        gats = []
        for _ in range(num_meta_paths):
            gats.append(GATConv(in_feat_size, out_size, num_heads, dropout, dropout, activation=ms.nn.ELU()))

        self.gats = ms.nn.CellList(gats)
        self.semantic = SemanticAttention(out_size * num_heads)
        self.num_meta_paths = num_meta_paths

    def construct(self, h, src_idx, dst_idx, n_nodes, n_edges):
        semantic_embeddings = []
        for i in range(self.num_meta_paths):
            semantic_embeddings.append(self.gats[i](h, src_idx[i], dst_idx[i], n_nodes[i], n_edges[i]))

        semantic_embeddings = ms.ops.Stack(1)(semantic_embeddings)
        ret = self.semantic(semantic_embeddings)
        return ret


class HAN(ms.nn.Cell):
    """ han """

    def __init__(self,
                 num_meta_paths,
                 in_feat_size,
                 hidden_size,
                 out_size,
                 num_heads,
                 dropout
                 ):
        super(HAN, self).__init__()
        layers = [HANLayer(num_meta_paths, in_feat_size, hidden_size, num_heads[0], dropout)]
        for i in range(1, len(num_heads)):
            layers.append(HANLayer(num_meta_paths, hidden_size * num_heads[i - 1], hidden_size, num_heads[i], dropout))
        self.layers = ms.nn.CellList(layers)
        self.predict = ms.nn.Dense(hidden_size * num_heads[-1], out_size)

    def construct(self, h, src_idx, dst_idx, n_nodes, n_edges):
        for conv in self.layers:
            h = conv(h, src_idx, dst_idx, n_nodes, n_edges)
        return self.predict(h)


class LossNet(ms.nn.Cell):
    """ loss net """

    def __init__(self, net):
        super(LossNet, self).__init__()
        self.net = net
        self.loss_fn = ms.nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    def construct(self, h, target, train_idx, src_idx, dst_idx, n_nodes, n_edges):
        pred = self.net(h, src_idx, dst_idx, n_nodes, n_edges)
        loss = self.loss_fn(pred[train_idx], target)
        return loss


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_han():
    """ test han """
    context.set_context(device_target="GPU", mode=context.GRAPH_MODE, enable_graph_kernel=True)
    hidden_size = 8
    num_heads = [8]
    drop_out = 0.2
    lr = 0.005
    weight_decay = 0.001
    epochs = 100
    npz = np.load(data_path)
    n_classes = int(npz["n_classes"])
    features = ms.Tensor(npz["features"], ms.float32)
    labels = ms.Tensor(npz["labels"], ms.int32)
    src_idx = [ms.Tensor(npz["pap_sid"], ms.int32), ms.Tensor(npz["plp_sid"], ms.int32)]
    dst_idx = [ms.Tensor(npz["pap_did"], ms.int32), ms.Tensor(npz["plp_did"], ms.int32)]
    num_nodes = int(npz['num_nodes'])
    n_classes = int(npz['n_classes'])
    feat_dim = int(npz['feat_dim'])
    train_idx = ms.Tensor(npz['train_idx'])
    test_idx = ms.Tensor(npz['test_idx'])
    labels = ms.Tensor(npz['labels'])
    train_labels = labels[train_idx]
    n_nodes = [num_nodes, num_nodes]  # Useless
    n_edges = ms.Tensor([-1, -1], ms.int32)  # Useless
    net = HAN(2, feat_dim, hidden_size, n_classes, num_heads, drop_out)
    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=lr, weight_decay=weight_decay)
    loss = LossNet(net)
    train_net = nn.TrainOneStepCell(loss, optimizer)
    warm_up = 3
    total = 0.
    for e in range(epochs):
        beg = time.time()
        train_net.set_train()
        train_net(features, train_labels, train_idx, src_idx, dst_idx, n_nodes, n_edges)

        end = time.time()
        dur = end - beg
        if e >= warm_up:
            total = total + dur

        net.set_train(False)
        out = net(features, src_idx, dst_idx, n_nodes, n_edges)
        test_predict = out[test_idx].asnumpy().argmax(axis=1)
        test_label = labels[test_idx].asnumpy()
        count = np.equal(test_predict, test_label)
        test_acc = np.sum(count) / test_label.shape[0]
    assert test_acc > 0.80
