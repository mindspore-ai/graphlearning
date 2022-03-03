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

""" example of training gin."""
import time
import argparse
import numpy as np
import mindspore as ms
from mindspore.profiler import Profiler
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.context as context
from mindspore.train.callback import TimeMonitor, LossMonitor

from mindspore_gl.nn.gnn_cell import GNNCell
from mindspore_gl.nn.conv import GINConv
from mindspore_gl.graph.ops import BatchHomoGraph, PadArray2d, PadHomoGraph, PadMode, PadDirection
from mindspore_gl.dataloader import RandomBatchSampler, Dataset, DataLoader
from mindspore_gl.dataset import IMDBBinary
from mindspore_gl import BatchedGraph, BatchedGraphField


class ApplyNodeFunc(nn.Cell):
    """
    Update the node feature hv with MLP, BN and ReLU.
    """

    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def construct(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = nn.ReLU()(h)
        return h


class MLP(nn.Cell):
    """MLP"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
        num_layers: number of layers in the neural networks (EXCLUDING the input layer).
            If num_layers=1, this reduces to linear model.
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden units at ALL layers
        output_dim: number of classes for prediction
        """
        super().__init__()
        self.num_layers = num_layers
        self.output_dim = output_dim
        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        if num_layers == 1:
            # Linear model
            self.linear = nn.Dense(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linears = nn.CellList()
            self.batch_norms = nn.CellList()

            self.linears.append(nn.Dense(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Dense(hidden_dim, hidden_dim))
            self.linears.append(nn.Dense(hidden_dim, output_dim))

            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def construct(self, x):
        if self.num_layers == 1:
            return self.linear(x)
        # If MLP
        h = x
        for layer in range(self.num_layers - 1):
            h = ms.ops.ReLU()(self.batch_norms[layer](self.linears[layer](h)))
        return self.linears[self.num_layers - 1](h)


class GinNet(GNNCell):
    """GinNet"""
    def __init__(self,
                 num_layers,
                 num_mlp_layers,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 final_dropout=0.1,
                 learn_eps=False,
                 graph_pooling_type='sum',
                 neighbor_pooling_type='sum'
                 ):
        super().__init__()
        self.final_dropout = final_dropout
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps

        self.mlps = nn.CellList()
        self.convs = nn.CellList()
        self.batch_norms = nn.CellList()

        if self.graph_pooling_type not in ("sum", "avg"):
            raise SyntaxError("Graph pooling type not supported yet.")
        for layer in range(num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.convs.append(GINConv(ApplyNodeFunc(self.mlps[layer]), learn_eps=self.learn_eps,
                                      aggregation_type=self.neighbor_pooling_type))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.linears_prediction = nn.CellList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Dense(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Dense(hidden_dim, output_dim))

    def construct(self, x, edge_weight, g: BatchedGraph):
        """GinNet forward"""
        hidden_rep = [x]
        h = x
        for layer in range(self.num_layers - 1):
            h = self.convs[layer](h, edge_weight, g)
            h = self.batch_norms[layer](h)
            h = nn.ReLU()(h)
            hidden_rep.append(h)

        score_over_layer = 0
        for layer, h in enumerate(hidden_rep):
            if self.graph_pooling_type == 'sum':
                pooled_h = g.sum_nodes(h)
            else:
                pooled_h = g.avg_nodes(h)
            score_over_layer = score_over_layer + nn.Dropout(self.final_dropout)(
                self.linears_prediction[layer](pooled_h))

        return score_over_layer


class LossNet(GNNCell):
    """
    LossNet definition
    """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')

    def construct(self, node_feat, edge_weight, target, g: BatchedGraph):
        predict = self.net(node_feat, edge_weight, g)
        target = ops.Squeeze()(target)
        loss = self.loss_fn(predict, target)
        ##############################
        # Mask Loss
        ##############################
        return ms.ops.ReduceSum()(loss * g.graph_mask)


class MultiHomoGraphDataset(Dataset):
    """MultiHomoGraphDataset"""
    def __init__(self, dataset, batch_size, mode=PadMode.CONST, node_size=1500, edge_size=15000):
        self._dataset = dataset
        self._batch_size = batch_size
        self.batch_fn = BatchHomoGraph()
        self.batched_edge_feat = None
        if mode == PadMode.CONST:
            self.node_feat_pad_op = PadArray2d(dtype=np.float32, mode=PadMode.CONST, direction=PadDirection.COL,
                                               size=(1500, dataset.num_features), fill_value=0)
            self.edge_feat_pad_op = PadArray2d(dtype=np.float32, mode=PadMode.CONST, direction=PadDirection.COL,
                                               size=(edge_size, dataset.num_edge_features), fill_value=0)
            self.graph_pad_op = PadHomoGraph(n_edge=edge_size, n_node=node_size, mode=PadMode.CONST)
        else:
            self.node_feat_pad_op = PadArray2d(dtype=np.float32, mode=PadMode.AUTO, direction=PadDirection.COL,
                                               fill_value=0)
            self.edge_feat_pad_op = PadArray2d(dtype=np.float32, mode=PadMode.AUTO, direction=PadDirection.COL,
                                               fill_value=0)
            self.graph_pad_op = PadHomoGraph(mode=PadMode.AUTO)
        ################
        # For Padding
        ################
        self.train_mask = np.array([True] * (self._batch_size + 1))
        self.train_mask[-1] = False

    def __getitem__(self, batch_graph_idx):
        graph_list = []
        feature_list = []
        for idx in range(batch_graph_idx.shape[0]):
            graph_list.append(self._dataset[batch_graph_idx[idx]])
            feature_list.append(self._dataset.graph_feat(batch_graph_idx[idx]))
        #########################
        # Batch Graph
        ########################
        batch_graph = self.batch_fn(graph_list)
        #########################
        # Pad Graph
        ########################
        batch_graph = self.graph_pad_op(batch_graph)
        #########################
        # Batch Node Feat
        ########################
        batched_node_feat = np.concatenate(feature_list)
        ###################
        # Pad NodeFeat
        ##################
        batched_node_feat = self.node_feat_pad_op(batched_node_feat)
        batched_label = self._dataset.graph_label[batch_graph_idx]
        ######################
        # Pad Label
        #####################
        batched_label = np.append(batched_label, batched_label[-1] * 0)
        #################################
        # Get Edge Feat
        #################################
        if self.batched_edge_feat is None or self.batched_edge_feat.shape[0] < batch_graph.edge_count:
            del self.batched_edge_feat
            self.batched_edge_feat = np.ones([batch_graph.edge_count, 1], dtype=np.float32)

        ##############################################################################
        # Trigger Node_Map_Idx/Edge_Map_Idx Computation, Because It Is Lazily Computed
        ###############################################################################
        _ = batch_graph.batch_meta.node_map_idx
        _ = batch_graph.batch_meta.edge_map_idx

        return batch_graph, batched_label, batched_node_feat, self.batched_edge_feat[:batch_graph.edge_count, :]


def main(train_args):
    if train_args.fuse:
        context.set_context(device_target="GPU", save_graphs=True, save_graphs_path="./computational_graph/",
                            mode=context.GRAPH_MODE, enable_graph_kernel=True)
    else:
        context.set_context(device_target="GPU")

    if train_args.profile:
        ms_profiler = Profiler(subgraph="ALL", is_detail=True, is_show_op_path=False, output_path="./prof_result")
    TimeMonitor()
    LossMonitor()
    #########################################
    # Set Up Dataset
    #########################################
    dataset = IMDBBinary(train_args.data_path)
    train_batch_sampler = RandomBatchSampler(dataset.train_graphs, batch_size=train_args.batch_size)
    multi_graph_dataset = MultiHomoGraphDataset(dataset, train_args.batch_size)
    train_dataloader = DataLoader(dataset=multi_graph_dataset, sampler=train_batch_sampler, num_workers=4,
                                  persistent_workers=True, prefetch_factor=4)

    test_batch_sampler = RandomBatchSampler(dataset.val_graphs, batch_size=train_args.batch_size)
    test_dataloader = DataLoader(dataset=multi_graph_dataset, sampler=test_batch_sampler, num_workers=0,
                                 persistent_workers=False)
    ###################################
    # Graph Mask
    ###################################
    np_graph_mask = [1] * (train_args.batch_size + 1)
    np_graph_mask[-1] = 0
    constant_graph_mask = ms.Tensor(np_graph_mask, dtype=ms.int32)

    #################################
    # Set Up Model
    ##################################
    net = GinNet(num_layers=train_args.num_layers,
                 num_mlp_layers=train_args.num_mlp_layers,
                 input_dim=dataset.num_features,
                 hidden_dim=train_args.hidden_dim,
                 output_dim=dataset.num_classes,
                 final_dropout=train_args.final_dropout,
                 learn_eps=train_args.learn_eps,
                 graph_pooling_type=train_args.graph_pooling_type,
                 neighbor_pooling_type=train_args.neighbor_pooling_type)

    learning_rates = nn.piecewise_constant_lr(
        [50, 100, 150, 200, 250, 300, 350], [0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003125, 0.00015625])
    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=learning_rates)
    loss = LossNet(net)
    train_net = nn.TrainOneStepCell(loss, optimizer)

    #########################################
    # Start To Train
    ########################################
    for epoch in range(train_args.epochs):
        start_time = time.time()
        train_net.set_train(True)
        train_loss = 0
        total_iter = 0
        total_time = 0.0
        for data in train_dataloader:
            batch_graph, label, node_feat, edge_feat = data
            # Create ms.Tensor
            node_feat = ms.Tensor.from_numpy(node_feat)
            edge_feat = ms.Tensor.from_numpy(edge_feat)
            label = ms.Tensor.from_numpy(label)
            batch_homo = BatchedGraphField(
                ms.Tensor.from_numpy(batch_graph.adj_coo[0]),
                ms.Tensor.from_numpy(batch_graph.adj_coo[1]),
                ms.Tensor(batch_graph.node_count, ms.int32),
                ms.Tensor(batch_graph.edge_count, ms.int32),
                ms.Tensor.from_numpy(batch_graph.batch_meta.node_map_idx),
                ms.Tensor.from_numpy(batch_graph.batch_meta.edge_map_idx),
                constant_graph_mask
            )
            # Train One Step
            train_loss += train_net(node_feat, edge_feat, label,
                                    *batch_homo.get_batched_graph()) / train_args.batch_size
            total_iter += 1
            if total_iter == train_args.iters_per_epoch:
                break

        train_loss /= train_args.iters_per_epoch
        end_time = time.time()
        total_time += end_time - start_time

        test_count = 0
        for data in test_dataloader:
            batch_graph, label, node_feat, edge_feat = data
            node_feat = ms.Tensor.from_numpy(node_feat)
            edge_feat = ms.Tensor.from_numpy(edge_feat)
            batch_homo = BatchedGraphField(
                ms.Tensor.from_numpy(batch_graph.adj_coo[0]),
                ms.Tensor.from_numpy(batch_graph.adj_coo[1]),
                ms.Tensor(batch_graph.node_count, ms.int32),
                ms.Tensor(batch_graph.edge_count, ms.int32),
                ms.Tensor.from_numpy(batch_graph.batch_meta.node_map_idx),
                ms.Tensor.from_numpy(batch_graph.batch_meta.edge_map_idx),
                constant_graph_mask
            )
            output = net(node_feat, edge_feat, *batch_homo.get_batched_graph()).asnumpy()
            label = label
            predict = np.argmax(output, axis=1)
            test_count += np.sum(np.equal(predict, label) * np_graph_mask)

        test_acc = test_count / len(test_dataloader) / train_args.batch_size
        print('Epoch {}, Time {:.3f} s, Train loss {},  Test acc {:.3f}'.format(epoch, end_time - start_time,
                                                                                train_loss, test_acc))
    print(f"check time per epoch {total_time / train_args.epochs}")
    ############################
    # Output Profiling Result
    ############################
    if train_args.profile:
        ms_profiler.analyse()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Graph convolutional neural net for whole-graph classification')
    parser.add_argument("--data_path", type=str, help="path to dataset")
    parser.add_argument("--dataset", type=str, default="IMDBBINARY", help="dataset")
    parser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    parser.add_argument('--epochs', type=int, default=350, help='number of epochs to train (default: 350)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64, help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5, help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "avg"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "avg", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true", default=True,
                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training '
                             'accuracy though.')
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument('--profile', type=bool, default=False, help="feature dimension")
    parser.add_argument('--fuse', type=bool, default=True, help="feature dimension")
    args = parser.parse_args()
    print(args)
    main(args)
