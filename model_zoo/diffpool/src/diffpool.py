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
"""Diffpool"""
import math
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.nn.probability.distribution as msd
from mindspore.common.initializer import XavierUniform

from mindspore_gl.nn import GNNCell
from mindspore_gl.nn import SAGEConv
from mindspore_gl import BatchedGraph
from mindspore_gl.parser.vcg import translate

def batch2tensor(batch_adj, batch_feat, node_per_pool_graph):
    """
    transform a batched graph to batched adjacency tensor and node feature tensor
    """
    batch_size = ops.Shape()(batch_adj)[0] // node_per_pool_graph
    adj_list = []
    feat_list = []
    for i in range(batch_size):
        start = i * node_per_pool_graph
        end = (i + 1) * node_per_pool_graph
        adj_list.append(ops.ExpandDims()(batch_adj[start:end, start:end], 0))
        feat_list.append(ops.ExpandDims()(batch_feat[start:end, :], 0))
    adj = ops.Concat(0)(adj_list)
    feat = ops.Concat(0)(feat_list)
    return feat, adj

class EntropyLoss(nn.Cell):
    """Entropy Loss"""
    def __init__(self):
        super().__init__()
        self.entropy_func = msd.Categorical().entropy

    def construct(self, s_l, node_mask):
        entropy_loss = self.entropy_func(s_l + 1e-8) * node_mask
        return ops.ReduceSum()(entropy_loss)


class LinkPredLoss(nn.Cell):
    """LinkPred Loss"""
    def construct(self, adj, s_l, ver_subgraph_idx=None, graph_mask=None):
        """construct function"""
        if len(ops.Shape()(adj)) == 3:
            link_pred_loss = adj - nn.MatMul()(s_l, ops.Transpose()(s_l, (0, 2, 1)))
            link_pred_loss = nn.Norm((1, 2))(link_pred_loss)
            link_pred_loss = link_pred_loss / (ops.Shape()(adj)[1] * ops.Shape()(adj)[2])
            return ops.ReduceMean()(link_pred_loss)
        node_mask = ops.Gather()(graph_mask, ver_subgraph_idx, 0)
        shape = (ops.Shape()(ver_subgraph_idx)[0], 1)
        node_mask = ops.Reshape()(node_mask, shape)
        scatter_ver_subgraph_idx = ops.Reshape()(ver_subgraph_idx, shape)
        num_of_nodes = ops.TensorScatterAdd()(
            ops.Zeros()((ops.Shape()(graph_mask)[0], 1), ms.int32),
            scatter_ver_subgraph_idx,
            node_mask
        ) + 0.0
        node_num = ops.MatMul(True)(num_of_nodes, num_of_nodes)
        link_pred_loss = adj - nn.MatMul(transpose_x2=True)(s_l, s_l)
        link_pred_loss = nn.Norm((1))(link_pred_loss)
        link_pred_loss = link_pred_loss / node_num
        return ops.ReduceMean()(link_pred_loss)


class DiffPoolBatchedGraphLayer(GNNCell):
    """DiffPool Batched Graph Layer"""
    def __init__(self, input_dim, assign_dim, output_feat_dim,
                 activation, aggregator_type, link_pred):
        super().__init__()
        self.embedding_dim = input_dim
        self.assign_dim = assign_dim
        self.hidden_dim = output_feat_dim
        self.link_pred = link_pred
        self.feat_gc = SAGEConv(
            in_feat_size=input_dim,
            out_feat_size=output_feat_dim,
            activation=activation,
            aggregator_type=aggregator_type)
        self.pool_gc = SAGEConv(
            in_feat_size=input_dim,
            out_feat_size=assign_dim,
            activation=activation,
            aggregator_type=aggregator_type)
        self.link_pred_loss = ms.Parameter(ms.Tensor(0, ms.float32), requires_grad=False)
        self.entropy_loss = ms.Parameter(ms.Tensor(0, ms.float32), requires_grad=False)

    def construct(self, h, g: BatchedGraph):
        """construct function"""
        feat = self.feat_gc(h, None, g)  # size = (sum_N, F_out), sum_N is num of nodes in this batch
        assign_tensor = self.pool_gc(h, None, g)  # size = (sum_N, N_a), N_a is num of nodes in pooled graph.
        assign_tensor = ops.Softmax(1)(assign_tensor)
        node_size, assign_size = ops.Shape()(assign_tensor)
        graph_size = ops.Shape()(g.graph_mask)[0]
        assign_tensor = ops.TensorScatterAdd()(
            ops.Zeros()((graph_size, node_size, assign_size), ms.float32),
            ops.Transpose()(
                ops.Stack()([g.ver_subgraph_idx, ms.nn.Range(0, node_size, 1)()]),
                (1, 0)
            ),
            assign_tensor
        )
        assign_tensor = ops.Transpose()(assign_tensor, (1, 0, 2))
        assign_tensor = ops.Reshape()(assign_tensor, (node_size, graph_size * assign_size))

        node_mask = ops.Gather()(g.graph_mask, g.ver_subgraph_idx, 0)
        assign_tensor = assign_tensor * ops.Reshape()(node_mask, (node_size, 1))
        h = nn.MatMul(transpose_x1=True)(assign_tensor, feat)
        # adj mm assign_tensor
        adj_new = ops.TensorScatterAdd()(
            ops.Zeros()((node_size, graph_size * assign_size), ms.float32),
            ops.Reshape()(g.dst_idx, (ops.Shape()(g.dst_idx)[0], 1)),
            ops.Gather()(assign_tensor, g.src_idx, 0)
        )
        adj_new = nn.MatMul(transpose_x1=True)(assign_tensor, adj_new)

        # adj to dense
        node_count = ops.Shape()(assign_tensor)[0]
        adj = ops.ScatterNd()(
            ms.ops.Transpose()(
                ms.ops.Stack()([g.src_idx, g.dst_idx]),
                (1, 0)
            ),
            ops.Ones()(ops.Shape()(g.src_idx), ms.int32),
            (node_count, node_count)
        )
        adj[-1][-1] = 0
        if self.link_pred:
            self.link_pred_loss = LinkPredLoss()(adj, assign_tensor, g.ver_subgraph_idx, g.graph_mask)
        self.entropy_loss = EntropyLoss()(assign_tensor, node_mask)

        return adj_new, h


class BatchedGraphSAGE(nn.Cell):
    """Batched GraphSAGE"""
    def __init__(self, infeat, outfeat, use_bn=False,
                 mean=False, add_self=False):
        super().__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.mean = mean
        gain = math.sqrt(2)  # gain for relu
        self.w = nn.Dense(infeat, outfeat, weight_init=XavierUniform(gain), has_bias=True)

    def construct(self, x, adj):
        """construct function"""
        num_node_per_graph = ops.Shape()(adj)[1]

        if self.add_self:
            adj = adj + ops.Eye()(num_node_per_graph, num_node_per_graph, ms.int32)

        if self.mean:
            adj = adj / ops.ReduceSum(True)(adj, -1)

        h_k_n = nn.MatMul()(adj, x)
        h_k = self.w(h_k_n)
        h_k = ops.L2Normalize(2)(h_k)
        h_k = ops.ReLU()(h_k)
        return h_k


class BatchedDiffPool(nn.Cell):
    """Batched DiffPool"""
    def __init__(self, nfeat, nnext, nhid, link_pred=False, entropy=True):
        super().__init__()
        self.link_pred = link_pred
        self.link_pred_layer = LinkPredLoss()
        self.embed = BatchedGraphSAGE(nfeat, nhid, use_bn=True)
        self.assign = BatchedGraphSAGE(nfeat, nnext, use_bn=True)
        self.reg_loss = nn.CellList()
        self.link_pred = link_pred
        self.entropy = entropy
        self.link_pred_loss = ms.Parameter(ms.Tensor(0, ms.float32), requires_grad=False)
        self.entropy_loss = ms.Parameter(ms.Tensor(0, ms.float32), requires_grad=False)

    def construct(self, x, adj):
        """construct function"""
        z_l = self.embed(x, adj)
        s_l = self.assign(x, adj)
        s_l = ops.Softmax(1)(s_l)
        xnext = nn.MatMul()(ops.Transpose()(s_l, (0, 2, 1)), z_l)
        anext = nn.MatMul()(ops.Transpose()(s_l, (0, 2, 1)), adj)
        anext = nn.MatMul()(anext, s_l)

        if self.link_pred:
            self.link_pred_loss = LinkPredLoss()(adj, s_l)
        if self.entropy:
            self.entropy_loss = EntropyLoss()(s_l)

        return xnext, anext


class DiffPool(GNNCell):
    """DiffPool"""
    def __init__(self, input_dim, hidden_dim, embedding_dim,
                 label_dim, activation, n_layers, n_pooling, linkpred, batch_size, aggregator_type,
                 assign_dim, pool_ratio, cat=False):
        super().__init__()
        self.link_pred = linkpred
        self.concat = cat
        self.n_pooling = n_pooling
        self.batch_size = batch_size

        self.gc_before_pool = nn.CellList()

        self.gc_after_pool = nn.CellList()
        self.assign_dim = assign_dim
        self.entropy = True
        self.bn = False
        self.num_aggs = 1

        translate(self, "gcn_construct")
        translate(self, "loss")

        assert n_layers >= 3, "n_layers too few"
        self.gc_before_pool.append(
            SAGEConv(
                in_feat_size=input_dim,
                out_feat_size=hidden_dim,
                activation=activation,
                aggregator_type=aggregator_type))
        for _ in range(n_layers - 2):
            self.gc_before_pool.append(
                SAGEConv(
                    in_feat_size=hidden_dim,
                    out_feat_size=hidden_dim,
                    activation=activation,
                    aggregator_type=aggregator_type))
        self.gc_before_pool.append(
            SAGEConv(
                in_feat_size=hidden_dim,
                out_feat_size=embedding_dim,
                activation=None,
                aggregator_type=aggregator_type))

        assign_dims = [self.assign_dim]
        if self.concat:
            pool_embedding_dim = hidden_dim * (n_layers - 1) + embedding_dim
        else:
            pool_embedding_dim = embedding_dim

        self.first_diffpool_layer = DiffPoolBatchedGraphLayer(
            pool_embedding_dim,
            self.assign_dim,
            hidden_dim,
            activation,
            aggregator_type,
            self.link_pred)
        gc_after_per_pool = nn.CellList()

        for _ in range(n_layers - 1):
            gc_after_per_pool.append(BatchedGraphSAGE(hidden_dim, hidden_dim))
        gc_after_per_pool.append(BatchedGraphSAGE(hidden_dim, embedding_dim))
        self.gc_after_pool.append(gc_after_per_pool)

        self.assign_dim = int(self.assign_dim * pool_ratio)
        if n_pooling == 2:
            self.second_diffpool_layer = BatchedDiffPool(
                pool_embedding_dim,
                self.assign_dim,
                hidden_dim,
                self.link_pred,
                self.entropy)
            gc_after_per_pool = nn.CellList()
            for _ in range(n_layers - 1):
                gc_after_per_pool.append(
                    BatchedGraphSAGE(
                        hidden_dim, hidden_dim))
            gc_after_per_pool.append(
                BatchedGraphSAGE(
                    hidden_dim, embedding_dim))
            self.gc_after_pool.append(gc_after_per_pool)
            assign_dims.append(self.assign_dim)
            self.assign_dim = int(self.assign_dim * pool_ratio)

        if self.concat:
            self.pred_input_dim = pool_embedding_dim * (n_pooling + 1) * self.num_aggs
        else:
            self.pred_input_dim = embedding_dim * self.num_aggs
        self.pred_layer = nn.Dense(self.pred_input_dim, label_dim, weight_init=XavierUniform(math.sqrt(2)))

    def gcn_construct(self, h, gc_layers, cat, g: BatchedGraph):
        """gcn construct function"""
        block_readout = []
        for gc_layer in gc_layers[:-1]:
            h = gc_layer(h, None, g)
            block_readout.append(h)
        h = gc_layers[-1](h, None, g)
        block_readout.append(h)
        if cat:
            block = ops.Concat(1)(block_readout)
        else:
            block = h
        return block

    def gcn_construct_tensorized(self, h, adj, gc_layers, cat=False):
        """gcn construct tensorized"""
        block_readout = []
        for gc_layer in gc_layers:
            h = gc_layer(h, adj)
            block_readout.append(h)
        if cat:
            block = ops.Concat(2)(block_readout)
        else:
            block = h
        return block

    def construct(self, h, g: BatchedGraph):
        """construct function"""
        out_all = []

        g_embedding = self.gcn_construct(h, self.gc_before_pool, self.concat, g)

        readout = g.max_nodes(g_embedding)
        out_all.append(readout)
        if self.num_aggs == 2:
            readout = g.sum_nodes(g_embedding)
            out_all.append(readout)

        adj, h = self.first_diffpool_layer(g_embedding, g)
        node_per_pool_graph = ops.Shape()(adj)[0] // ops.Shape()(g.graph_mask)[0]

        h, adj = batch2tensor(adj, h, node_per_pool_graph)

        h = self.gcn_construct_tensorized(
            h, adj, self.gc_after_pool[0], self.concat)
        readout = ops.ReduceMax()(h, 1)
        out_all.append(readout)
        if self.num_aggs == 2:
            readout = ops.ReduceSum()(h, 1)
            out_all.append(readout)

        if self.n_pooling == 2:
            h, adj = self.second_diffpool_layer(h, adj)
            h = self.gcn_construct_tensorized(
                h, adj, self.gc_after_pool[1], self.concat)
            readout = ops.ReduceMax()(h, 1)
            out_all.append(readout)
            if self.num_aggs == 2:
                readout = ops.ReduceSum()(h, 1)
                out_all.append(readout)

        if self.concat:
            final_readout = ops.Concat(1)(out_all)
        else:
            final_readout = readout
        ypred = self.pred_layer(final_readout)
        return ypred

    def loss(self, pred, label, g: BatchedGraph):
        """loss"""
        criterion = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')
        loss = criterion(pred, label)
        loss = ops.ReduceMean()(loss * g.graph_mask)

        if self.n_pooling == 2:
            if self.link_pred:
                loss = loss + self.second_diffpool_layer.link_pred_loss
            if self.entropy:
                loss = loss + self.second_diffpool_layer.entropy_loss

        if self.link_pred:
            loss = loss + self.first_diffpool_layer.link_pred_loss
        return loss
