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
"""Train embedding"""
import time
import argparse
import numpy as np
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.context as context

from mindspore_gl.dataloader import RandomBatchSampler
from mindspore_gl.dataset import BlogCatalog

from src.deepwalk import BatchRandWalk, SkipGramModel, DeepWalkDataset


def main(arguments):
    if arguments.save_file_path is None:
        arguments.save_file_path = arguments.data_path
        if arguments.save_file_path[-1] != '/':
            arguments.save_file_path = arguments.save_file_path + '/'

    if arguments.fuse and arguments.device == "GPU":
        context.set_context(device_target=arguments.device, save_graphs=True, save_graphs_path="./computational_graph/",
                            mode=context.GRAPH_MODE, enable_graph_kernel=True, device_id=arguments.device_id)
    else:
        context.set_context(device_target=arguments.device, device_id=arguments.device_id, mode=context.GRAPH_MODE)

    dataset = BlogCatalog(arguments.data_path)
    n_nodes = dataset.node_count

    batch_fn = BatchRandWalk(
        graph=dataset[0],
        walk_len=arguments.walk_len,
        win_size=arguments.win_size,
        neg_num=arguments.neg_num,
        batch_size=arguments.batch_size)

    sampler = RandomBatchSampler(
        data_source=[i for i in range(n_nodes * arguments.epoch)],
        batch_size=arguments.batch_size)

    deepwalk_dataset = DeepWalkDataset(
        nodes=[i for i in range(n_nodes)],
        batch_fn=batch_fn,
        repeat=arguments.epoch,
        length=len(list(sampler))
    )

    net = SkipGramModel(
        num_nodes=n_nodes + 1,  # for padding
        embed_size=arguments.embed_size,
        neg_num=arguments.neg_num)

    dataloader = ds.GeneratorDataset(deepwalk_dataset, ['src', 'dsts', 'node_mask', 'pair_count'],
                                     sampler=sampler, python_multiprocessing=True, max_rowsize=9)

    lrs = []
    data_len = len(sampler)
    lr = arguments.lr
    end_lr = 0.0001
    reduce_per_iter = (lr - end_lr) / data_len
    for _ in range(data_len):
        lrs.append(lr)
        lr -= reduce_per_iter
    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=lrs)
    train_net = nn.TrainOneStepCell(net, optimizer)

    print("Totally {} iterations".format(data_len))
    train_net.set_train(True)
    before_train = time.time()
    for curr_iter, data in enumerate(dataloader):
        src, dsts, node_mask, pair_count = data
        # Since we use padding, we should not use reduceMean cause we mul with a node_mask when computing loss.
        # So here we should calculate the loss divided by pair_count.
        loss = train_net(src, dsts, node_mask) / pair_count
        after_train = time.time()
        print("Iter {}, train loss: {}, time: {:.4f}".format(curr_iter, loss, after_train - before_train))
        before_train = time.time()

    # save the embedding weight
    embedding_weight = net.s_emb.embedding_table.asnumpy()
    if arguments.norm:
        embedding_weight /= np.sqrt(np.sum(embedding_weight * embedding_weight, 1)).reshape(-1, 1)
    np.savez(arguments.save_file_path + arguments.save_file_name, embedding_weight=embedding_weight)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deepwalk')
    parser.add_argument("--data_path", type=str, help="path to dataloader")
    parser.add_argument("--device", type=str, default="GPU", help="which device to use")
    parser.add_argument("--device_id", type=int, default=0, help="which device id to use")
    parser.add_argument("--save_file_path", type=str, default=None,
                        help="path to save embedding weight. If None, data_path will be used.")
    parser.add_argument("--save_file_name", type=str, default="deepwalk_embedding.npz",
                        help="file name of saved embedding weight")
    parser.add_argument("--dataset", type=str, default="BlogCatalog", help="dataset")
    parser.add_argument("--epoch", type=int, default=80, help="number of epoch")
    parser.add_argument("--lr", type=float, default=0.025, help='learning rate')
    parser.add_argument("--batch_size", type=int, default=128, help="number of batch size")
    parser.add_argument("--neg_num", type=int, default=20, help="number of negative sampling")
    parser.add_argument("--walk_len", type=int, default=40, help="walk length")
    parser.add_argument("--win_size", type=int, default=10, help="window size")
    parser.add_argument("--embed_size", type=int, default=128, help="size of embedding")
    parser.add_argument("--norm", type=bool, default=True, help="whether do norm")
    parser.add_argument('--fuse', type=bool, default=True, help="whether to use graph mode")
    args = parser.parse_args()
    print(args)
    main(args)
