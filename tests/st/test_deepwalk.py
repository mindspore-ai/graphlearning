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
""" test deepwalk """
import os
import numpy as np
import pytest
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.context as context
from mindspore.common.initializer import Uniform

os.environ["LD_LIBRARY_PATH"] += ":/lib:/usr/lib:/usr/local/lib"

from mindspore_gl.dataset import BlogCatalog
import mindspore_gl.sample_kernel as sample_kernel
from mindspore_gl.sampling import random_walk_unbias_on_homo
from mindspore_gl.dataloader import RandomBatchSampler, Dataset, DataLoader
from sklearn.metrics import f1_score

data_path = "/home/workspace/mindspore_dataset/GNN_Dataset/"


class SkipGramModel(nn.Cell):
    """SkipGramModel"""

    def __init__(self,
                 num_nodes,
                 embed_size=16,
                 neg_num=5):
        super(SkipGramModel, self).__init__()

        self.num_nodes = num_nodes
        self.neg_num = neg_num

        self.s_emb = nn.Embedding(num_nodes, embed_size, embedding_table=Uniform(0.5 / embed_size))
        self.d_emb = nn.Embedding(num_nodes, embed_size, embedding_table=Uniform(0))

        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.sum = ops.ReduceSum()
        self.matmul = nn.MatMul(transpose_x2=True)

    def construct(self, src, dsts, node_mask):
        """SkipGramModel construct"""
        src_embed = self.s_emb(src)
        dsts_embed = self.d_emb(dsts)

        pos_embed = dsts_embed[:, 0:1]
        neg_embed = dsts_embed[:, 1:]

        pos_logits = self.matmul(src_embed, pos_embed)  # [total pair count, 1, 1]

        neg_logits = self.matmul(src_embed, neg_embed)  # [total pair count, 1, neg_num]

        ones_label = ops.ones_like(pos_logits)
        pos_loss = self.loss(pos_logits, ones_label)
        pos_loss = self.sum(pos_loss * node_mask)

        zeros_label = ops.zeros_like(neg_logits)
        neg_loss = self.loss(neg_logits, zeros_label)
        neg_loss = self.sum(self.sum(neg_loss * node_mask, 2))

        loss = (pos_loss + neg_loss) / 2
        return loss


class BatchRandWalk:
    """BatchRandWalk"""

    def __init__(self, graph, walk_len, win_size, neg_num, batch_size):
        self.graph = graph
        self.walk_len = walk_len - 1  # include head node itself
        self.win_size = win_size
        self.neg_num = neg_num
        self.batch_size = batch_size
        self.padded_size = batch_size * walk_len * win_size * 2
        self.fill_value = self.graph.node_count

    def __call__(self, nodes):
        walks = random_walk_unbias_on_homo(self.graph, np.array(nodes, np.int32), self.walk_len)
        src_list, pos_list = [], []
        for walk in walks:
            s, p = sample_kernel.skip_gram_gen_pair(walk, self.win_size)
            src_list.append(s)
            pos_list.append(p)
            src_list.append(p)
            pos_list.append(s)
        src = [s for x in src_list for s in x]
        pos = [s for x in pos_list for s in x]
        pair_count = len(src)

        # negative sampling
        negs = np.random.randint(low=0, high=self.graph.node_count, size=[pair_count, self.neg_num]).tolist()
        for i in range(pair_count):
            while src[i] in negs[i] or pos[i] in negs[i]:
                negs[i] = np.random.randint(low=0, high=self.graph.node_count, size=[self.neg_num]).tolist()

        src = np.array(src, dtype=np.int32)
        pos = np.array(pos, dtype=np.int32)
        negs = np.array(negs, dtype=np.int32)
        src, pos = np.reshape(src, [-1, 1]), np.reshape(pos, [-1, 1])
        dsts = np.concatenate([pos, negs], 1)

        # here we padding with fill_value = n_nodes
        node_mask = np.concatenate(
            [np.ones((pair_count), np.int32), np.zeros((self.padded_size - pair_count), np.int32)])
        src = np.concatenate([src, np.ones((self.padded_size - pair_count, 1), np.int32) * self.fill_value])
        dsts = np.concatenate(
            [dsts, np.ones((self.padded_size - pair_count, 1 + self.neg_num), np.int32) * self.fill_value])
        node_mask = np.reshape(node_mask, [-1, 1, 1])

        return src, dsts, node_mask, pair_count


class DeepWalkDataset(Dataset):
    """DeepWalkDataset"""

    def __init__(self, nodes, batch_fn: BatchRandWalk, repeat=1):
        self.repeat = repeat
        self.data = nodes
        self.datalen = len(nodes)
        self.batch_fn = batch_fn

    def __getitem__(self, batch_idxs):
        return self.batch_fn([self.data[idx % self.datalen] for idx in batch_idxs])

    def __len__(self):
        return len(self.data) * self.repeat


class Model(nn.Cell):
    """Model"""

    def __init__(self, embed_size, num_classes):
        super(Model, self).__init__()
        self.dense = nn.Dense(embed_size, num_classes)

    def construct(self, node_emb):
        logits = self.dense(node_emb)
        return logits


class LossNet(nn.Cell):
    """LossNet"""

    def __init__(self, net):
        super(LossNet, self).__init__()
        self.net = net
        self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)

    def construct(self, node_emb, label):
        """LossNet construct"""
        logits = self.net(node_emb)
        loss = self.loss_fn(logits, label)
        loss = ops.ReduceMean()(loss)
        return loss


class PredictDataset(Dataset):
    """PredictDataset"""

    def __init__(self, nodes, label):
        self.data = nodes
        self.label = label
        self.datalen = len(nodes)

    def __getitem__(self, batch_idxs):
        return np.take(self.data, batch_idxs, 0), np.take(self.label, batch_idxs, 0)

    def __len__(self):
        return self.datalen


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_deepwalk():
    """test deepwalk"""
    context.set_context(device_target="GPU", mode=context.GRAPH_MODE, enable_graph_kernel=True)

    epochs = 40
    walk_len = 20
    win_size = 10

    embed_size = 128
    neg_num = 10
    batch_size = 128
    dataset = BlogCatalog(data_path)
    n_nodes = dataset.node_count

    batch_fn = BatchRandWalk(
        graph=dataset[0],
        walk_len=walk_len,
        win_size=win_size,
        neg_num=neg_num,
        batch_size=batch_size)

    deepwalk_dataset = DeepWalkDataset(
        nodes=[i for i in range(n_nodes)],
        batch_fn=batch_fn,
        repeat=epochs)

    net = SkipGramModel(
        num_nodes=n_nodes + 1,  # for padding
        embed_size=embed_size,
        neg_num=neg_num)

    sampler = RandomBatchSampler(
        data_source=[i for i in range(n_nodes * epochs)],
        batch_size=batch_size)

    dataloader = DataLoader(
        dataset=deepwalk_dataset,
        sampler=sampler,
        num_workers=10,
        persistent_workers=True)

    lrs = []
    data_len = len(dataloader)
    lr = 0.025
    end_lr = 0.0001
    reduce_per_iter = (lr - end_lr) / data_len
    for _ in range(data_len):
        lrs.append(lr)
        lr -= reduce_per_iter
    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=lrs)
    train_net = nn.TrainOneStepCell(net, optimizer)

    train_net.set_train(True)
    for _, data in enumerate(dataloader):
        src, dsts, node_mask, pair_count = data
        src = ms.Tensor.from_numpy(src)
        dsts = ms.Tensor.from_numpy(dsts)
        node_mask = ms.Tensor.from_numpy(node_mask)
        res = train_net(src, dsts, node_mask) / pair_count
        print(res)

    # norm the embedding weight
    embedding_weight = net.s_emb.embedding_table.asnumpy()
    embedding_weight /= np.sqrt(np.sum(embedding_weight * embedding_weight, 1)).reshape(-1, 1)

    batch_size = 32
    embedding_table = ms.Parameter(ms.Tensor(embedding_weight, ms.float32), requires_grad=False)
    emb = nn.Embedding(
        ops.Shape()(embedding_table)[0],
        embed_size,
        embedding_table=embedding_table)

    vocab = dataset.vocab
    label = dataset.node_label

    node_feat = emb(ms.Tensor(vocab, ms.int32)).asnumpy()

    vocab_size = len(label)
    np.random.seed(2)
    mask = np.random.permutation(vocab_size)
    train_mask, test_mask = mask[:int(vocab_size * 0.9)], mask[int(vocab_size * 0.9):]

    predict_dataset = PredictDataset(node_feat, label)

    train_sampler = RandomBatchSampler(
        data_source=train_mask,
        batch_size=batch_size)

    train_dataloader = DataLoader(
        dataset=predict_dataset,
        sampler=train_sampler,
        num_workers=0,
        persistent_workers=False)

    test_sampler = RandomBatchSampler(
        data_source=test_mask,
        batch_size=batch_size)

    test_dataloader = DataLoader(
        dataset=predict_dataset,
        sampler=test_sampler,
        num_workers=0,
        persistent_workers=False)

    net = Model(embed_size, dataset.num_classes)
    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=0.001)
    loss = LossNet(net)
    train_net = nn.TrainOneStepCell(loss, optimizer)

    for _ in range(epochs):
        train_net.set_train(True)
        train_loss = 0
        for data in train_dataloader:
            node_feat, label = data
            node_feat, label = ms.Tensor.from_numpy(node_feat), ms.Tensor.from_numpy(label)
            train_loss += train_net(node_feat, label).asnumpy()
        train_loss /= len(train_dataloader)

    train_net.set_train(False)
    test_pred, test_label = [], []
    for data in test_dataloader:
        node_feat, label = data
        node_feat = ms.Tensor.from_numpy(node_feat)
        logits = net(node_feat).asnumpy()
        pred = np.argmax(logits, axis=1)
        test_label.extend(label.tolist())
        test_pred.extend(pred.tolist())
    test_macro_f1 = f1_score(test_label, test_pred, average='macro')

    assert test_macro_f1 > 0.23
