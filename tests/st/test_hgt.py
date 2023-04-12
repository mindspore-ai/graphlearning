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
""" test hgt """
import math
from typing import List, Tuple
import time
import pytest
import numpy as np

import mindspore as ms
import mindspore.context as context
import mindspore.ops.functional as F
from mindspore.common.initializer import initializer
from mindspore.common.initializer import XavierUniform

from mindspore_gl import Graph
from mindspore_gl.nn import GNNCell

data_path = "/home/workspace/mindspore_dataset/GNN_Dataset/acm_with_mask.npz"


class HomoHGTLayer(GNNCell):
    """homo HGT layer"""

    def __init__(self, n_heads, d_k,
                 k_cell, q_cell, v_cell):
        super().__init__()
        gain = math.sqrt(2)
        self.pri = ms.Parameter(ms.ops.Ones()((n_heads, 1), ms.float32))
        self.msg = ms.Parameter(initializer(XavierUniform(gain), [n_heads, d_k * d_k], ms.float32), name="relation_msg")
        self.att = ms.Parameter(initializer(XavierUniform(gain), [n_heads, d_k * d_k], ms.float32), name="relation_att")
        self.n_heads = n_heads
        self.d_k = d_k
        self.sqrt_dk = math.sqrt(d_k)
        self.kc = k_cell
        self.qc = q_cell
        self.vc = v_cell
        self.exp = ms.ops.Exp()
        self.reduce = ms.ops.ReduceMin()

    def construct(self, src_x, x, g: Graph):
        """homo HGT layer forward"""
        k = ms.ops.Reshape()(self.kc(src_x), (-1, self.n_heads, self.d_k))
        v = ms.ops.Reshape()(self.vc(src_x), (-1, self.n_heads, self.d_k))
        q = ms.ops.Reshape()(self.qc(x), (-1, self.n_heads, self.d_k))
        k_tran = ms.ops.Transpose()(ms.ops.BatchMatMul()(ms.ops.Transpose()(k, (1, 0, 2)),
                                                         ms.ops.Reshape()(self.att, (-1, self.d_k, self.d_k))),
                                    (1, 0, 2))
        v_tran = ms.ops.Transpose()(ms.ops.BatchMatMul()(ms.ops.Transpose()(v, (1, 0, 2)),
                                                         ms.ops.Reshape()(self.msg, (-1, self.d_k, self.d_k))),
                                    (1, 0, 2))
        g.set_vertex_attr({"qe": q, "ke": k_tran, "ve": v_tran})
        for v in g.dst_vertex:
            e = [ms.ops.Exp()(ms.ops.ReduceSum(keep_dims=True)(v.qe * u.ke, -1) * self.pri / self.sqrt_dk) for u in
                 v.innbs]
            attn_score = [c / g.sum(e) for c in e]
            a = [u.ve for u in v.innbs]
            v.ret = g.sum(attn_score * a)
        ret = [v.ret for v in g.dst_vertex]
        return ret


class HeteroHGTLayer(ms.nn.Cell):
    """Hetero HGT layer"""

    def __init__(self,
                 num_node_types: int,
                 num_edge_types: int,
                 canonical_etypes: List[Tuple],
                 hidden_size: int,
                 output_size: int,
                 dropout: float = 0.8,
                 n_heads: int = 4,
                 use_norm=True) -> None:
        super().__init__()
        self.num_ntypes = num_node_types
        self.num_etypes = num_edge_types
        self.canoical_etypes = canonical_etypes
        self.output_size = output_size
        self.use_norm = use_norm
        cl_k_tmp = []
        cl_q_tmp = []
        cl_v_tmp = []
        cl_a_tmp = []
        if use_norm:
            cl_norm_tmp = []
        for i in range(num_node_types):
            cl_k_tmp.append(ms.nn.Dense(hidden_size, output_size))
            cl_q_tmp.append(ms.nn.Dense(hidden_size, output_size))
            cl_v_tmp.append(ms.nn.Dense(hidden_size, output_size))
            cl_a_tmp.append(ms.nn.Dense(output_size, output_size))
            if use_norm:
                cl_norm_tmp.append(ms.nn.LayerNorm((output_size,)))
        cl_k = ms.nn.CellList(cl_k_tmp)
        cl_q = ms.nn.CellList(cl_q_tmp)
        cl_v = ms.nn.CellList(cl_v_tmp)
        self.cl_a = ms.nn.CellList(cl_a_tmp)
        self.skip = ms.Parameter(ms.ops.Ones()((num_node_types,), ms.float32), name="skip{}".format(i))
        if use_norm:
            self.cl_norm = ms.nn.CellList(cl_norm_tmp)
        d_k = output_size // n_heads
        self.drop = ms.nn.Dropout(p=dropout)
        layer = []
        for stype, _, dtype in canonical_etypes:
            layer.append(HomoHGTLayer(n_heads, d_k, cl_k[stype], cl_q[dtype], cl_v[stype]))
        self.layers = ms.nn.CellList(layer)

    def construct(self, h, src_idx, dst_idx, n_nodes, n_edges):
        """Hetero HGT layer forward"""
        out = []
        count = []
        for i in range(self.num_ntypes):
            out.append(ms.ops.Zeros()((1,), ms.float32))
            count.append(0)
        for src_type, etype, dst_type in self.canoical_etypes:
            out[dst_type] += self.layers[etype](h[src_type], h[dst_type], src_idx[etype], dst_idx[etype],
                                                n_nodes[etype], n_edges[etype])
            count[dst_type] += 1
        for i in range(self.num_ntypes):
            out[i] = out[i] / count[i]

        new_h = []
        for ntype in range(self.num_ntypes):
            alpha = ms.ops.Sigmoid()(self.skip[ntype])
            t = ms.ops.Reshape()(out[ntype], (-1, self.output_size))
            emb = self.cl_a[ntype](t)
            dropped = self.drop(emb)
            trans_out = dropped * alpha + h[ntype] * (1 - alpha)
            if self.use_norm:
                new_h.append(self.cl_norm[ntype](trans_out))
            else:
                new_h.append(trans_out)
        return new_h


class HGT(ms.nn.Cell):
    """HGT net"""

    def __init__(self,
                 num_node_types: int,
                 num_edge_types: int,
                 canonical_etypes: List[Tuple],
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 dropout: float = 0.8,
                 n_layers: int = 2,
                 n_heads: int = 4,
                 use_norm=True) -> None:
        super().__init__()
        self.num_ntypes = num_node_types
        self.num_etypes = num_edge_types
        self.canoical_etypes = canonical_etypes
        cl = []
        for _ in range(num_node_types):
            cl.append(ms.nn.Dense(input_size, hidden_size))
        self.cl = ms.nn.CellList(cl)

        layers = []
        for _ in range(n_layers):
            layers.append(
                HeteroHGTLayer(num_node_types, num_edge_types, canonical_etypes, hidden_size, hidden_size, dropout,
                               n_heads, use_norm))
        self.layers = ms.nn.CellList(layers)
        self.out = ms.nn.Dense(hidden_size, output_size)

    def construct(self, h, out_id, src_idx, dst_idx, n_nodes, n_edges):
        """HGT Net forward"""
        new_h = []
        for i in range(self.num_ntypes):
            new_h.append(ms.ops.GeLU()(self.cl[i](h[i])))
        for i in range(len(self.layers)):
            new_h = self.layers[i](new_h, src_idx, dst_idx, n_nodes, n_edges)
        return self.out(new_h[out_id])


class LossNet(ms.nn.Cell):
    """loss definition"""

    def __init__(self, net) -> None:
        super().__init__()
        self.net = net
        self.loss_fn = ms.nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    def construct(self, h, target, train_idx, out_id, src_idx, dst_idx, n_nodes, n_edges):
        """HGT net with loss function"""
        predict = self.net(h, out_id, src_idx, dst_idx, n_nodes, n_edges)
        loss = self.loss_fn(predict[train_idx], target)
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

    def construct(self, h, target, train_idx, out_id, src_idx, dst_idx, n_nodes, n_edges):
        """one step train with forward and backward"""
        weights = self.weights
        loss = self.network(h, target, train_idx, out_id, src_idx, dst_idx, n_nodes, n_edges)
        grads = self.grad(self.network, weights)(h, target, train_idx, out_id, src_idx, dst_idx, n_nodes, n_edges, 1.0)
        grads = self.hyper_map(F.partial(clip_grad, 1.0), grads)
        grads = self.grad_reducer(grads)
        succ = self.optimizer(grads)
        return F.depend(loss, succ)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_hgt():
    """test HGT net"""
    context.set_context(device_target="GPU", mode=context.GRAPH_MODE, enable_graph_kernel=True)
    epochs = 20
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
    n_nodes = [num_a_nodes, num_p_nodes, num_p_nodes, num_p_nodes, num_l_nodes, num_p_nodes]
    n_edges = ms.Tensor([-1, -1, -1, -1, -1, -1], ms.int32)  # Useless
    net = HGT(num_node_types=3, num_edge_types=len(cannonical_etypes), canonical_etypes=cannonical_etypes,
              input_size=input_size, hidden_size=hidden_size, output_size=n_classes)
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
        train_net(h_tensor, train_labels, train_idx, 0, src_idx, dst_idx, n_nodes, n_edges)
        end = time.time()
        dur = end - beg
        if e >= warm_up:
            total = total + dur

        net.set_train(False)
        out = net(h_tensor, 0, src_idx, dst_idx, n_nodes, n_edges)
        test_predict = out[test_idx].asnumpy().argmax(axis=1)
        test_label = labels[test_idx].asnumpy()
        count = np.equal(test_predict, test_label)
        test_acc = np.sum(count) / test_label.shape[0]
    assert test_acc > 0.38
