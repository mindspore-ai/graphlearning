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
"""HSGNN model implemented using mindspore-gl"""
import time
import random
import argparse
import numpy as np
import mindspore as ms
import mindspore.context as context
from mindspore.profiler import Profiler
from src.datasets import Texas, multilyer_rwsampling
from src.models import HSGNN, LossNet
from src.utils import random_splits
import seaborn as sns


def set_seed(seed):
    """ set seed """
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


def metrics(out, label, test_mask):
    """ metrics computation """
    res_mask = ms.Tensor(test_mask)
    test_out = out[res_mask].asnumpy()
    predict = np.argmax(test_out, axis=1)
    test_acc = np.equal(predict, label[test_mask]).sum() / len(test_mask)

    return test_acc


def run_exp(args, train_net, dataset, net):
    """train procedure"""
    best_val_acc = 0
    best_test_acc = 0
    best_acc = 0.
    time_run = []
    val_accs = []
    test_accs = []

    for epoch in range(args.epochs):
        t_st = time.time()
        train_net.set_train(True)
        x = ms.Tensor.from_numpy(dataset.x).astype(ms.float32)
        edge_index = ms.Tensor.from_numpy(dataset.edge_index)
        y = ms.Tensor.from_numpy(dataset.y)
        train_mask = ms.Tensor(dataset.train_mask)
        train_phase = ms.Tensor(1, dtype=ms.int32)

        starts, ends, end_indexs = multilyer_rwsampling(dataset[0], args,
                                                        args.nlayer)
        inputs = [x, edge_index, y, starts, ends, end_indexs, train_mask,
                  train_phase]
        train_loss = train_net(*inputs)

        time_epoch = time.time() - t_st  # each epoch train times
        time_run.append(time_epoch)

        # for evaluation
        train_net.set_train(False)
        train_phase = ms.Tensor(0, dtype=ms.int32)
        out = net(x, edge_index, starts, ends, end_indexs, train_phase)

        val_acc = metrics(out, dataset.y, dataset.val_mask)
        test_acc = metrics(out, dataset.y, dataset.test_mask)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        if epoch % args.show_step == 0:
            one_step_info = "RunTime-Epoch: {0} | Train - loss: {1} | " \
                            "Val - acc: {2:.4f} | Test - acc: {3:.4f} - " \
                            "best: {4:.4f}/{5:.4f} | epoch time: {6:.2f} ms " \
                .format(epoch, train_loss, np.mean(val_accs[-args.show_step:]),
                        np.mean(test_accs[-args.show_step:]), best_test_acc,
                        best_acc,
                        np.mean(time_run[-args.show_step:]) * 1000)
            print(one_step_info)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_acc = test_acc

        if test_acc > best_test_acc:
            best_test_acc = test_acc
    return best_val_acc, best_acc, time_run


def main(train_args):
    """main procedure"""
    context.set_context(device_target='GPU', mode=context.GRAPH_MODE,
                        save_graphs=True,
                        save_graphs_path="./saved_ir/")

    set_seed(train_args.seed)
    dataset = Texas(train_args.data_path)

    percls_trn = int(
        round(train_args.train_rate * len(dataset.y) / dataset.num_classes))
    val_lb = int(round(train_args.val_rate * len(dataset.y)))

    if train_args.profile:
        ms_profiler = Profiler(subgraph="ALL", is_detail=True,
                               is_show_op_path=False, output_path="log")

    net = HSGNN(dataset, train_args)
    dataset = random_splits(dataset, dataset.num_classes, percls_trn, val_lb,
                            train_args.seed)

    optimizer = ms.nn.optim.Adam(net.trainable_params(),
                                 learning_rate=train_args.lr,
                                 weight_decay=train_args.weight_decay)
    loss = LossNet(net)
    train_net = ms.nn.TrainOneStepCell(loss, optimizer)
    best_val_acc, best_acc, time_run = run_exp(train_args,
                                               train_net,
                                               dataset,
                                               net)

    epoch_time = time_run[100:]
    uncertainty_time = np.max(np.abs(sns.utils.ci(
        sns.algorithms.bootstrap(epoch_time, func=np.mean, n_boot=1000),
        95) - np.asarray(epoch_time).mean()))

    print(
        f'test acc mean = {best_acc * 100:.4f}  \t val acc mean = \
        {best_val_acc:.4f}')
    print(
        f'avg epoch time = {np.mean(epoch_time) * 1000:.4f} ms ± \
        {uncertainty_time * 100 * 1000:.4f} ')

    if train_args.profile:
        ms_profiler.analyse()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HSGNN")
    parser.add_argument('--data-path', type=str, default='./data/',
                        help="path to dataloader")
    parser.add_argument('--seed', type=int, default=4023022221,
                        help='seeds for random splits.')
    parser.add_argument('--epochs', type=int, default=1000, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='weight decay.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden units.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout for neural networks.')
    parser.add_argument('--rws', type=int, default=10,
                        help='random walks per node')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='vertices per batch')
    parser.add_argument('--train_rate', type=float, default=0.6,
                        help='train set rate.')
    parser.add_argument('--val_rate', type=float, default=0.2,
                        help='val set rate.')
    parser.add_argument('--k', type=int, default=2, help='propagation steps.')
    parser.add_argument('--nlayer', type=int, default=2,
                        help='num of network layers')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='alpha for APPN.')
    parser.add_argument('--dprate', type=float, default=0.9,
                        help='dropout for propagation layer.')
    parser.add_argument('--show_step', type=int, default=50,
                        help='show stpe info')
    parser.add_argument('--profile', type=bool, default=False,
                        help="feature dimension")
    argument = parser.parse_args()
    print(argument)
    main(argument)
