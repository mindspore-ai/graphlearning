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
""" test rgcn """
import math
import time
from typing import List
import pytest
import numpy as np

import mindspore as ms
from mindspore.common.initializer import initializer
from mindspore.common.initializer import XavierUniform
import mindspore.context as context
import mindspore.ops.functional as F

from mindspore_gl.nn import GNNCell
from mindspore_gl import Graph

data_path = "/home/workspace/mindspore_dataset/GNN_Dataset/acm_with_mask.npz"


class HomoRGCNConv(GNNCell):
    """homo rgcn conv"""

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 activation: callable = None):
        super().__init__()
        gain = math.sqrt(2)
        self.w = ms.Parameter(initializer(XavierUniform(gain), [in_size, out_size], ms.float32), name="w")
        self.act = activation

    def construct(self, x, in_deg, g: Graph):
        """homo rgcn conv forward"""
        g.set_vertex_attr(
            {"d": ms.ops.Reshape()(in_deg, ms.ops.Shape()(in_deg) + (1,)), "h": ms.ops.MatMul()(x, self.w)})
        for v in g.dst_vertex:
            v.h = g.sum([u.h for u in v.innbs]) / v.d
        ret = [v.h for v in g.dst_vertex]
        if self.act is not None:
            ret = self.act(ret)
        return ret


class RGCN(ms.nn.Cell):
    """Rgcn net"""

    def __init__(self,
                 num_node_types: int,
                 cannonical_etypes: List[int],
                 input_size: int,
                 hidden_size: int,
                 output_size: int) -> None:
        super().__init__()
        self.can_etypes = cannonical_etypes
        self.n_types = num_node_types
        self.layer1 = HomoRGCNConv(input_size, hidden_size, ms.nn.LeakyReLU(0.01))
        self.layer2 = HomoRGCNConv(hidden_size, output_size, None)

    def construct(self, h, out_id, in_deg, src_idx, dst_idx, n_nodes, n_edges):
        """rgcn net forward"""
        new_h = []
        out = []
        for _ in range(self.n_types):
            new_h.append(ms.ops.Zeros()((1,), ms.float32))
            out.append(ms.ops.Zeros()((1,), ms.float32))
        for src_type, edge_type, dst_type in self.can_etypes:
            new_h[dst_type] += self.layer1(h[src_type], in_deg[edge_type], src_idx[edge_type], dst_idx[edge_type],
                                           n_nodes[edge_type], n_edges[edge_type])
        for src_type, edge_type, dst_type in self.can_etypes:
            out[dst_type] += self.layer2(new_h[src_type], in_deg[edge_type], src_idx[edge_type], dst_idx[edge_type],
                                         n_nodes[edge_type], n_edges[edge_type])
        return out[out_id]


class LossNet(ms.nn.Cell):
    """loss definition"""

    def __init__(self, net) -> None:
        super().__init__()
        self.net = net
        self.loss_fn = ms.nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    def construct(self, h, target, train_idx, out_id, in_deg, src_idx, dst_idx, n_nodes, n_edges):
        """rgcn net with loss"""
        predict = self.net(h, out_id, in_deg, src_idx, dst_idx, n_nodes, n_edges)
        loss = self.loss_fn(predict[train_idx], target)
        # if np.isnan(loss.asnumpy()):
        #    raise RuntimeError("Found NAN", h[out_id], predict[train_idx], target)
        return loss


clip_grad = ms.ops.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Tensor")
def _clip_grad(clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    dt = ms.ops.dtype(grad)
    new_grad = ms.nn.ClipByNorm()(grad, ms.ops.cast(ms.ops.tuple_to_array((clip_value,)), dt))
    return new_grad


class TrainOneStepCellWithGradClipping(ms.nn.TrainOneStepCell):
    """one step train cell"""

    def __init__(self, net, optimizer, clip_val: float = 1.0) -> None:
        super().__init__(net, optimizer)
        self.clip = clip_val
        self.hyper_map = ms.ops.HyperMap()

    def construct(self, h, target, train_idx, out_id, in_deg, src_idx, dst_idx, n_nodes, n_edges):
        """one step train with forward and backward"""
        weights = self.weights
        loss = self.network(h, target, train_idx, out_id, in_deg, src_idx, dst_idx, n_nodes, n_edges)
        grads = self.grad(self.network, weights)(h, target, train_idx, out_id, in_deg, src_idx, dst_idx, n_nodes,
                                                 n_edges, 1.0)
        grads = self.hyper_map(F.partial(clip_grad, 1.0), grads)
        grads = self.grad_reducer(grads)
        succ = self.optimizer(grads)
        return F.depend(loss, succ)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_rgcn():
    """test rgcn net"""
    context.set_context(device_target="GPU", mode=context.GRAPH_MODE, enable_graph_kernel=True)
    epochs = 100
    hidden_size = 256
    input_size = 256
    npz = np.load(data_path)
    cannonical_etypes = [(0, 0, 1), (1, 1, 0), (0, 2, 0), (0, 3, 0), (0, 4, 2), (2, 5, 0)]
    src_idx = [ms.Tensor(npz["pva_sid"], ms.int32), ms.Tensor(npz["pva_trans_sid"], ms.int32),
               ms.Tensor(npz["pvp_sid"], ms.int32),
               ms.Tensor(npz["pvp_trans_sid"], ms.int32), ms.Tensor(npz["pvl_sid"], ms.int32),
               ms.Tensor(npz["pvl_trans_sid"], ms.int32)]
    dst_idx = [ms.Tensor(npz["pva_did"], ms.int32), ms.Tensor(npz["pva_trans_did"], ms.int32),
               ms.Tensor(npz["pvp_did"], ms.int32),
               ms.Tensor(npz["pvp_trans_did"], ms.int32), ms.Tensor(npz["pvl_did"], ms.int32),
               ms.Tensor(npz["pvl_trans_did"], ms.int32)]
    num_a_nodes = int(npz['num_a_nodes'])
    num_l_nodes = int(npz['num_l_nodes'])
    num_p_nodes = int(npz['num_p_nodes'])
    n_nodes = [num_a_nodes, num_p_nodes, num_p_nodes, num_p_nodes, num_l_nodes, num_p_nodes]
    in_deg = [ms.ops.TensorScatterAdd()(ms.ops.Ones()((n,), ms.int32),  # avoid divide by zero
                                        ms.ops.Reshape()(idx, ms.ops.Shape()(idx) + (1,)),
                                        ms.ops.OnesLike()(idx)) for n, idx in zip(n_nodes, dst_idx)]

    n_classes = int(npz['n_classes'])
    train_idx = ms.Tensor(npz['train_idx'])
    test_idx = ms.Tensor(npz['test_idx'])
    labels = ms.Tensor(npz['labels'])
    train_labels = labels[train_idx]
    gain = math.sqrt(2)
    h = [ms.Tensor(init=XavierUniform(gain), shape=(num_p_nodes, input_size), dtype=ms.float32).asnumpy(),
         ms.Tensor(init=XavierUniform(gain), shape=(num_a_nodes, input_size), dtype=ms.float32).asnumpy(),
         ms.Tensor(init=XavierUniform(gain), shape=(num_l_nodes, input_size), dtype=ms.float32).asnumpy()]
    h_tensor = [ms.Tensor(v, dtype=ms.float32) for v in h]
    n_edges = ms.Tensor([-1, -1, -1, -1, -1, -1], ms.int32)  # Useless
    net = RGCN(num_node_types=3, cannonical_etypes=cannonical_etypes, input_size=input_size, hidden_size=hidden_size,
               output_size=n_classes)
    loss = LossNet(net)
    # lr_sched = ms.nn.piecewise_constant_lr()
    # Add gradient clipping
    optimizer = ms.nn.optim.AdamWeightDecay(net.trainable_params(), weight_decay=0.01, eps=1e-8)
    train_net = ms.nn.TrainOneStepCell(loss, optimizer)
    total = 0.
    warm_up = 3
    for e in range(epochs):
        beg = time.time()
        train_net.set_train()
        train_net(h_tensor, train_labels, train_idx, 0, in_deg, src_idx, dst_idx, n_nodes, n_edges)
        end = time.time()
        dur = end - beg
        if e >= warm_up:
            total = total + dur

        net.set_train(False)
        out = net(h_tensor, 0, in_deg, src_idx, dst_idx, n_nodes, n_edges)
        test_predict = out[test_idx].asnumpy().argmax(axis=1)
        test_label = labels[test_idx].asnumpy()
        count = np.equal(test_predict, test_label)
        test_acc = np.sum(count) / test_label.shape[0]
    assert test_acc > 0.37
