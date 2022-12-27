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
"""Train eval"""
import math
import argparse
import time
import numpy as np
import scipy.io as sio
import mindspore as ms
import mindspore.context as context
import mindspore.ops.functional as F
from mindspore.common.initializer import XavierUniform
from mindspore_gl import HeterGraphField
from src.hgt import HGT


class LossNet(ms.nn.Cell):
    """Loss Net"""
    def __init__(self, net) -> None:
        super().__init__()
        self.net = net
        self.loss_fn = ms.nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    def construct(self, h, target, train_idx, out_id, src_idx, dst_idx, n_nodes, n_edges):
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
    """Train one step cell with grad clipping"""
    def __init__(self, net, optimizer, clip_val: float = 1.0) -> None:
        super().__init__(net, optimizer)
        self.clip = clip_val
        self.hyper_map = ms.ops.HyperMap()

    def construct(self, h, target, train_idx, out_id, src_idx, dst_idx, n_nodes, n_edges):
        weights = self.weights
        loss = self.network(h, target, train_idx, out_id, src_idx, dst_idx, n_nodes, n_edges)
        grads = self.grad(self.network, weights)(h, target, train_idx, out_id, src_idx, dst_idx, n_nodes, n_edges, 1.0)
        grads = self.hyper_map(F.partial(clip_grad, 1.0), grads)
        grads = self.grad_reducer(grads)
        succ = self.optimizer(grads)
        return F.depend(loss, succ)


def load_data_acm(data_path='/your/path/to/ACM.mat'):
    """preprocess for ACM datasets"""
    data = sio.loadmat(data_path)
    src_idx = []
    dst_idx = []
    edge_types = ['PvsA', 'PvsA_trans', 'PvsP', 'PvsP_trans', 'PvsL', 'PvsL_trans']
    num_nodes = {'P': 0, 'A': 0, 'L': 0}
    for idx in range(len(edge_types)):
        e_type = edge_types[idx]
        if e_type[-5:] == 'trans':
            e_data = data[e_type[:-6]].transpose()
            e_data = e_data.tocoo()
        else:
            e_data = data[e_type]
            e_data = e_data.tocoo()
            num_nodes[e_type[0]] = max(num_nodes[e_type[0]], max(e_data.row))
            num_nodes[e_type[3]] = max(num_nodes[e_type[3]], max(e_data.col))
        src_idx.append(ms.Tensor(list(e_data.row), ms.int32))
        dst_idx.append(ms.Tensor(list(e_data.col), ms.int32))
    cannonical_etypes = [(0, 0, 1), (1, 1, 0), (0, 2, 0), (0, 3, 0), (0, 4, 2), (2, 5, 0)]
    pvc = data['PvsC'].tocsr()
    p_selected = pvc.tocoo()
    labels = pvc.indices
    pid = p_selected.row
    return src_idx, dst_idx, pid, labels, num_nodes, cannonical_etypes


def main(arguments):
    context.set_context(device_target="GPU", mode=context.PYNATIVE_MODE)
    if arguments.fuse:
        context.set_context(device_target="GPU", mode=context.GRAPH_MODE, enable_graph_kernel=True)

    src_idx, dst_idx, pid, labels, num_nodes, cannonical_etypes = load_data_acm(arguments.data_path)
    num_a_nodes = int(num_nodes['A']) + 1
    num_l_nodes = int(num_nodes['L']) + 1
    num_p_nodes = int(num_nodes['P']) + 1
    n_classes = int(max(labels)) + 1
    labels = ms.Tensor(labels)
    shuffle = np.random.permutation(pid)
    train_idx = ms.Tensor(shuffle[0:800])
    test_idx = ms.Tensor(shuffle[900:])
    train_labels = labels[train_idx]
    cannonical_etypes = [(0, 0, 1), (1, 1, 0), (0, 2, 0), (0, 3, 0), (0, 4, 2), (2, 5, 0)]

    gain = math.sqrt(2)
    h = [ms.Tensor(init=XavierUniform(gain), shape=(num_p_nodes, arguments.input_size), dtype=ms.float32).asnumpy(),
         ms.Tensor(init=XavierUniform(gain), shape=(num_a_nodes, arguments.input_size), dtype=ms.float32).asnumpy(),
         ms.Tensor(init=XavierUniform(gain), shape=(num_l_nodes, arguments.input_size), dtype=ms.float32).asnumpy()]
    h_tensor = [ms.Tensor(v, dtype=ms.float32) for v in h]
    n_nodes = [num_a_nodes, num_p_nodes, num_p_nodes, num_p_nodes, num_l_nodes, num_p_nodes]
    n_edges = []
    for edges in src_idx:
        n_edges.append(ms.ops.Shape()(edges)[0])
    hgf = HeterGraphField(src_idx, dst_idx, n_nodes, n_edges)
    net = HGT(num_node_types=3, num_edge_types=len(cannonical_etypes), canonical_etypes=cannonical_etypes,
              input_size=arguments.input_size, hidden_size=arguments.hidden_size, output_size=n_classes)
    loss = LossNet(net)
    # lr_sched = ms.nn.piecewise_constant_lr()
    # Add gradient clipping
    optimizer = ms.nn.optim.AdamWeightDecay(net.trainable_params(), weight_decay=0.01, eps=1e-8)
    train_net = ms.nn.TrainOneStepCell(loss, optimizer)
    total = 0.
    warm_up = 3
    for e in range(arguments.epochs):
        beg = time.time()
        train_net.set_train()
        train_loss = train_net(h_tensor, train_labels, train_idx, 0, *hgf.get_heter_graph())
        end = time.time()
        dur = end - beg
        if e >= warm_up:
            total = total + dur

        net.set_train(False)
        out = net(h_tensor, 0, *hgf.get_heter_graph())
        test_predict = out[test_idx].asnumpy().argmax(axis=1)
        test_label = labels[test_idx].asnumpy()
        count = np.equal(test_predict, test_label)
        print('Epoch:{} Epoch time:{} ms Train loss {} Test acc:{}'.format(e, dur * 1000, train_loss,
                                                                           np.sum(count) / test_label.shape[0]))
    print("Model:{} Dataset:{} Avg epoch time:{}".format("HGT", arguments.data_path, total * 1000
                                                         / (arguments.epochs - warm_up)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HGT")
    parser.add_argument('--data_path', type=str, help='Path to dataset', required=True)
    parser.add_argument('--epochs', type=int, help='Number of epochs to train', default=200)
    parser.add_argument('--hidden_size', type=int, help='Hidden size per node', default=256)
    parser.add_argument('--input_size', type=int, help='Input size per node', default=256)
    parser.add_argument('--clip', type=float, help='Gradient clip value', default=1.)
    parser.add_argument('--max-lr', type=float, help='Max learning rate', default=1e-3)
    parser.add_argument('--fuse', type=bool, default=False, help="enable fusion")
    args = parser.parse_args()
    print(args)
    main(args)
