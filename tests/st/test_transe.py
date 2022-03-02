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
""" test transe """
import math
import os
import random
import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Uniform
import mindspore.context as context

from mindspore_gl.dataloader.dataloader import DataLoader
from mindspore_gl.dataloader.dataset import Dataset
from mindspore_gl.dataloader.samplers import RandomBatchSampler

data_path = "/home/workspace/mindspore_dataset/GNN_Dataset/FB15k"


class TransE(nn.Cell):
    """ TransE definition """

    def __init__(self,
                 n_entity,
                 n_relation,
                 embedding_dim,
                 margin_value,
                 score_func,
                 batch_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.margin_value = margin_value
        self.score_func = score_func
        self.batch_size = batch_size
        self.n_entity = n_entity

        if score_func not in ('L1', 'L2'):
            raise SyntaxError("Score function should be L1 or L2.")

        bound = 6 / math.sqrt(self.embedding_dim)
        self.e_emb = nn.Embedding(n_entity, embedding_dim, embedding_table=Uniform(bound))
        self.r_emb = nn.Embedding(n_relation, embedding_dim, embedding_table=Uniform(bound))
        r_embedding = ops.L2Normalize()(self.r_emb.embedding_table)
        self.r_emb = nn.Embedding(n_relation, embedding_dim, embedding_table=r_embedding)
        self.margin = ms.Tensor([self.margin_value for _ in range(self.batch_size)], ms.float32)

        self.reducesum = ops.ReduceSum()
        self.abs = ops.Abs()
        self.square = ops.Square()
        self.relu = nn.ReLU()
        self.topk = ops.TopK()

    def construct(self, head_pos, tail_pos, relation_pos, head_neg, tail_neg, relation_neg):
        """ TransE construct """
        head_pos_emb = self.e_emb(head_pos)
        tail_pos_emb = self.e_emb(tail_pos)
        relation_pos_emb = self.r_emb(relation_pos)
        head_neg_emb = self.e_emb(head_neg)
        tail_neg_emb = self.e_emb(tail_neg)
        relation_neg_emb = self.r_emb(relation_neg)

        distance_pos = head_pos_emb + relation_pos_emb - tail_pos_emb
        distance_neg = head_neg_emb + relation_neg_emb - tail_neg_emb

        if self.score_func == 'L1':
            score_pos = self.reducesum(self.abs(distance_pos), 1)
            score_neg = self.reducesum(self.abs(distance_neg), 1)
        else:
            score_pos = self.reducesum(self.square(distance_pos), 1)
            score_neg = self.reducesum(self.square(distance_neg), 1)
        loss = self.reducesum(self.relu(self.margin + score_pos - score_neg))
        return loss


class KnowLedgeGraphDataset:
    """ KnowLedgeGraphDataset definition """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.entity_dict = {}
        self.entities = []
        self.relation_dict = {}
        self.n_entity = 0
        self.n_relation = 0
        self.train_mask = []
        self.test_mask = []
        self.val_mask = []
        # load triples
        self.load_dicts()
        self.training_triples, self.n_training_triple = self.load_triples('train')
        self.validation_triples, self.n_validation_triple = self.load_triples('valid')
        self.test_triples, self.n_test_triple = self.load_triples('test')
        self.triples = self.training_triples + self.validation_triples + self.test_triples
        # generate triple pools
        self.training_triple_pool = set(self.training_triples)
        self.triple_pool = set(self.triples)
        # generate masks
        self.generate_mask()

    def load_dicts(self):
        """ KnowLedgeGraphDataset load dicts """
        with open(os.path.join(self.data_dir, 'entity2id.txt'), "r") as f:
            e_k, e_v = [], []
            for line in f.readlines():
                info = line.strip().replace("\n", "").split("\t")
                e_k.append(info[0])
                e_v.append(int(info[1]))
        self.entity_dict = dict(zip(e_k, e_v))
        self.n_entity = len(self.entity_dict)
        self.entities = list(self.entity_dict.values())
        with open(os.path.join(self.data_dir, 'relation2id.txt'), "r") as f:
            r_k, r_v = [], []
            for line in f.readlines():
                info = line.strip().replace("\n", "").split("\t")
                r_k.append(info[0])
                r_v.append(int(info[1]))
        self.relation_dict = dict(zip(r_k, r_v))
        self.n_relation = len(self.relation_dict)

    def load_triples(self, mode):
        """ KnowLedgeGraphDataset load triples """
        assert mode in ('train', 'valid', 'test')
        with open(os.path.join(self.data_dir, mode + '.txt'), "r") as f:
            hs, ts, rs = [], [], []
            for line in f.readlines():
                info = line.strip().replace("\n", "").split("\t")
                hs.append(info[0])
                ts.append(info[1])
                rs.append(info[2])
        triples = list(zip([self.entity_dict[h] for h in hs],
                           [self.entity_dict[t] for t in ts],
                           [self.relation_dict[r] for r in rs]))
        n_triple = len(triples)
        return triples, n_triple

    def generate_mask(self):
        """ KnowLedgeGraphDataset generate mask """
        self.train_mask = np.arange(0, self.n_training_triple)
        self.val_mask = np.arange(self.n_training_triple, self.n_training_triple + self.n_validation_triple)
        self.test_mask = np.arange(self.n_training_triple + self.n_validation_triple,
                                   self.n_training_triple + self.n_validation_triple + self.n_test_triple)


class TrainDataset(Dataset):
    """ TrainDataset definition """

    def __init__(self, kg: KnowLedgeGraphDataset):
        super().__init__()
        self.kg = kg

    def __getitem__(self, batch_idxs):
        tri_pos = [self.kg.triples[idx] for idx in batch_idxs]
        head_pos, tail_pos, relation_pos = [], [], []
        head_neg, tail_neg, relation_neg = [], [], []
        for h, t, r in tri_pos:
            head_pos.append(h)
            tail_pos.append(t)
            relation_pos.append(r)
            h_neg = h
            t_neg = t
            head_prob = np.random.binomial(1, 0.5)
            while True:
                if head_prob:
                    h_neg = random.choice(self.kg.entities)
                else:
                    t_neg = random.choice(self.kg.entities)
                if (h_neg, t_neg, r) not in self.kg.training_triple_pool:
                    break
            head_neg.append(h_neg)
            tail_neg.append(t_neg)
            relation_neg.append(r)
        return np.array(head_pos, np.int32), np.array(tail_pos, np.int32), np.array(relation_pos, np.int32), \
               np.array(head_neg, np.int32), np.array(tail_neg, np.int32), np.array(relation_neg, np.int32)


e_emb = None
r_emb = None


class EvalDataset(Dataset):
    """ EvalDataset definition """

    def __init__(self, kg: KnowLedgeGraphDataset, score_func):
        super().__init__()
        self.kg = kg
        self.score_func = score_func

    def __getitem__(self, idx):
        idx = idx[0]
        h, t, r = self.kg.triples[idx]
        global e_emb, r_emb
        return evaluate(e_emb, r_emb, h, t, r, self.score_func, self.kg)


def get_rank(kg, head_position, tail_position, head, tail, relation):
    """ get rank """
    head_rank_raw = 0
    tail_rank_raw = 0
    head_rank_filter = 0
    tail_rank_filter = 0
    for candidate in head_position:
        if candidate == head:
            break
        head_rank_raw += 1
        if (candidate, tail, relation) not in kg.triple_pool:
            head_rank_filter += 1
    for candidate in tail_position:
        if candidate == tail:
            break
        tail_rank_raw += 1
        if (head, candidate, relation) not in kg.triple_pool:
            tail_rank_filter += 1
    return head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter


def evaluate(e_embedding, r_embedding, head, tail, relation, score_func, kg):
    """ evaluate """
    head_emb = e_embedding[head]
    tail_emb = e_embedding[tail]
    relation_emb = r_embedding[relation]

    distance_head_prediction = e_embedding + relation_emb - tail_emb
    distance_tail_prediction = head_emb + relation_emb - e_embedding

    if score_func == 'L1':
        idx_head_prediction = np.argsort(np.sum(np.abs(distance_head_prediction), axis=1))
        idx_tail_prediction = np.argsort(np.sum(np.abs(distance_tail_prediction), axis=1))
    else:
        idx_head_prediction = np.argsort(np.sum(np.abs(distance_head_prediction), axis=1))
        idx_tail_prediction = np.argsort(np.sum(np.abs(distance_tail_prediction), axis=1))

    return get_rank(kg, idx_head_prediction, idx_tail_prediction, head, tail, relation)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_transe():
    """ test transe """
    context.set_context(device_target="GPU", mode=context.GRAPH_MODE, enable_graph_kernel=True)

    batch_size = 10000
    embedding_dim = 100
    score_func = 'L1'
    lr = 0.003
    workers = 24
    margin_value = 1.0
    max_epoch = 100

    kg = KnowLedgeGraphDataset(data_path)
    train_dataset = TrainDataset(kg)
    train_batch_sampler = RandomBatchSampler(kg.train_mask, batch_size=batch_size)
    train_dataloader = DataLoader(dataset=train_dataset, sampler=train_batch_sampler, num_workers=workers,
                                  persistent_workers=True)
    eval_dataset = EvalDataset(kg, score_func)
    eval_batch_sampler = RandomBatchSampler(kg.test_mask, batch_size=1)
    eval_dataloader = DataLoader(dataset=eval_dataset, sampler=eval_batch_sampler, num_workers=workers,
                                 persistent_workers=False)

    net = TransE(n_entity=kg.n_entity,
                 n_relation=kg.n_relation,
                 embedding_dim=embedding_dim,
                 margin_value=margin_value,
                 score_func=score_func,
                 batch_size=batch_size)
    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=lr)
    train_net = nn.TrainOneStepCell(net, optimizer)

    l2_normalize = ms.ops.L2Normalize(axis=1)
    for _ in range(1, max_epoch + 1):
        for data in train_dataloader:
            train_net.set_train(False)
            # better result if this line removed
            net.e_emb.embedding_table = l2_normalize(net.e_emb.embedding_table)
            train_net.set_train(True)
            head_pos, tail_pos, relation_pos, head_neg, tail_neg, relation_neg = data
            head_pos = ms.Tensor.from_numpy(head_pos)
            tail_pos = ms.Tensor.from_numpy(tail_pos)
            relation_pos = ms.Tensor.from_numpy(relation_pos)
            head_neg = ms.Tensor.from_numpy(head_neg)
            tail_neg = ms.Tensor.from_numpy(tail_neg)
            relation_neg = ms.Tensor.from_numpy(relation_neg)
            train_net(head_pos, tail_pos, relation_pos, head_neg, tail_neg, relation_neg)

    train_net.set_train(False)

    head_meanrank_raw = 0
    head_hits10_raw = 0
    tail_meanrank_raw = 0
    tail_hits10_raw = 0

    head_meanrank_filter = 0
    head_hits10_filter = 0
    tail_meanrank_filter = 0
    tail_hits10_filter = 0

    global e_emb, r_emb
    e_emb = net.e_emb.embedding_table.asnumpy()
    r_emb = net.r_emb.embedding_table.asnumpy()
    for _, data in enumerate(eval_dataloader):
        head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter = data
        head_meanrank_raw += head_rank_raw
        if head_rank_raw < 10:
            head_hits10_raw += 1
        tail_meanrank_raw += tail_rank_raw
        if tail_rank_raw < 10:
            tail_hits10_raw += 1
        head_meanrank_filter += head_rank_filter
        if head_rank_filter < 10:
            head_hits10_filter += 1
        tail_meanrank_filter += tail_rank_filter
        if tail_rank_filter < 10:
            tail_hits10_filter += 1
    head_meanrank_raw /= len(eval_dataloader)
    head_hits10_raw /= len(eval_dataloader)
    tail_meanrank_raw /= len(eval_dataloader)
    tail_hits10_raw /= len(eval_dataloader)
    meanrank_raw = (head_meanrank_raw + tail_meanrank_raw) / 2
    hits10_raw = (head_hits10_raw + tail_hits10_raw) / 2
    assert meanrank_raw < 300
    assert hits10_raw > 0.34

    head_meanrank_filter /= len(eval_dataloader)
    head_hits10_filter /= len(eval_dataloader)
    tail_meanrank_filter /= len(eval_dataloader)
    tail_hits10_filter /= len(eval_dataloader)
    meanrank_filter = (head_meanrank_filter + tail_meanrank_filter) / 2
    hits10_filter = (head_hits10_filter + tail_hits10_filter) / 2
    assert meanrank_filter < 160
    assert hits10_filter > 0.47
