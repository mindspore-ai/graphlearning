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
"""models"""
import numpy as np
import mindspore as ms
from mindspore.common.initializer import XavierUniform
from mindspore_gl import Graph
from mindspore_gl.nn import GNNCell


class HSGNN(ms.nn.Cell):
    """HSGNN Model"""

    def __init__(self, dataset, args):
        super().__init__()
        self.nlayer = args.nlayer
        self.num_nodes = dataset.num_nodes
        self.num_edges = dataset.num_edges
        hidden = args.hidden
        module_list = [ms.nn.Dense(dataset.node_feat_size * 2, hidden,
                                   weight_init=XavierUniform())]
        for _ in range(args.nlayer - 1):
            module_list.append(
                ms.nn.Dense(hidden * 2, hidden, weight_init=XavierUniform()))
        module_list.append(ms.nn.Dense(hidden, dataset.num_classes))
        self.lins = ms.nn.CellList(module_list)
        self.samconvs = ms.nn.CellList(
            [SampAggConv(dataset, args) for i in range(args.nlayer)])
        self.dropout = ms.nn.Dropout(p=args.dropout)
        self.dprate = ms.nn.Dropout(p=args.dprate)
        self.relu = ms.nn.ReLU()
        self.num_classes = dataset.num_classes

    def construct(self, x, edge_index, starts, ends, end_indexs, train_phase):
        """HSGNN forward"""
        for i in range(self.nlayer):
            x = self.samconvs[i](x, train_phase, starts[i], ends[i],
                                 end_indexs[i], edge_index[0], edge_index[1],
                                 self.num_nodes, self.num_edges)
            x = self.relu(self.lins[i](self.dprate(x)))
        x = self.lins[-1](self.dropout(x))
        return x


class SampAggConv(GNNCell):
    """SampAggConv Net"""

    def __init__(self, dataset, args):
        super().__init__()
        samp_w = args.alpha * ((1 - args.alpha) ** np.arange(0, args.k + 1))
        samp_w[-1] = (1 - args.alpha) ** args.k
        self.samp_w = ms.Parameter(samp_w.astype(np.float32),
                                   requires_grad=True)
        self.graph = dataset[0]
        self.degree = ms.Tensor(dataset.indegree, dtype=ms.float32)
        self.norm = ms.Tensor((self.degree ** -0.5).reshape(-1, 1)).astype(
            ms.float32)  # (node_num, 1)
        self.isolated_nodes_mask = ms.Tensor.from_numpy(dataset.isolated_nodes)
        self.n = dataset.num_nodes
        self.k = args.k
        self.rws = args.rws

    def construct(self, x, train_phase, starts, ends, end_index, g: Graph):
        """SampAggConv Net forward"""
        if train_phase:
            batch_num = self.n
            rws_1 = self.rws * (self.k + 1)
            aggx = (ms.ops.ExpandDims()(
                ((self.degree[starts] ** -0.5) * (self.degree[ends] ** -0.5) *
                 self.samp_w[end_index] / self.rws), 1) * x[ends]).reshape(
                     batch_num, rws_1, x.shape[-1]).sum(axis=1)
            if self.isolated_nodes_mask.shape[0]:
                aggx[self.isolated_nodes_mask] *= 0
            aggx = ms.ops.Concat(axis=1)((x, aggx))
        else:
            aggx = x * self.samp_w[0]
            for k in range(self.k):
                x = x * self.norm
                g.set_vertex_attr({"x": x})
                for v in g.dst_vertex:
                    v.x = g.sum([u.x for u in v.innbs])
                x = [v.x for v in g.dst_vertex] * self.norm
                aggx += x * self.samp_w[k + 1]
            if self.isolated_nodes_mask.shape[0]:
                aggx[self.isolated_nodes_mask] *= 0
            aggx = ms.ops.Concat(axis=1)((x, aggx))
        return aggx


class LossNet(ms.nn.Cell):
    """ LossNet definition """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = ms.nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True,
                                                                reduction='none'
                                                                )

    def construct(self, x, edge_index, y, starts, ends, end_indexs, mask,
                  train_phase):
        out = self.net(x, edge_index, starts, ends, end_indexs, train_phase)[
            mask]
        loss = self.loss_fn(out, y[mask])
        return ms.ops.ReduceSum()(loss) / len(y)
