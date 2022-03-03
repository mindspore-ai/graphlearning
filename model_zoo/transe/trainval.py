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
"""train eval"""
import argparse
import os
import time
import random
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.context as context

from mindspore_gl.dataloader import RandomBatchSampler, Dataset, DataLoader
from src.transe import TransE


class KnowLedgeGraphDataset:
    """Knowledge Graph Dataset"""
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
        """Load dicts"""
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
        """Load triples"""
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
        """generate mask"""
        self.train_mask = np.arange(0, self.n_training_triple)
        self.val_mask = np.arange(self.n_training_triple, self.n_training_triple + self.n_validation_triple)
        self.test_mask = np.arange(self.n_training_triple + self.n_validation_triple,
                                   self.n_training_triple + self.n_validation_triple + self.n_test_triple)


class TrainDataset(Dataset):
    """train dataset"""
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
    """Eval Dataset"""
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
    """Get rank"""
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
    """Evaluate"""
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


def main(arguments):
    if arguments.fuse:
        context.set_context(device_target="GPU", device_id=7, mode=context.GRAPH_MODE, enable_graph_kernel=True)
    else:
        context.set_context(device_target="GPU")

    kg = KnowLedgeGraphDataset(arguments.data_path)
    train_dataset = TrainDataset(kg)
    train_batch_sampler = RandomBatchSampler(kg.train_mask, batch_size=arguments.batch_size)
    train_dataloader = DataLoader(dataset=train_dataset, sampler=train_batch_sampler, num_workers=arguments.workers,
                                  persistent_workers=True)
    eval_dataset = EvalDataset(kg, arguments.score_func)
    eval_batch_sampler = RandomBatchSampler(kg.test_mask, batch_size=1)
    eval_dataloader = DataLoader(dataset=eval_dataset, sampler=eval_batch_sampler, num_workers=arguments.workers,
                                 persistent_workers=False)

    net = TransE(n_entity=kg.n_entity,
                 n_relation=kg.n_relation,
                 embedding_dim=arguments.embedding_dim,
                 margin_value=arguments.margin_value,
                 score_func=arguments.score_func,
                 batch_size=arguments.batch_size)
    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=arguments.lr)
    train_net = nn.TrainOneStepCell(net, optimizer)

    l2_normalize = ms.ops.L2Normalize(axis=1)
    for e in range(1, arguments.max_epoch + 1):
        before_train = time.time()
        train_loss = 0
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
            train_loss += train_net(head_pos, tail_pos, relation_pos, head_neg, tail_neg, relation_neg)
        train_loss /= arguments.batch_size * len(train_dataloader)
        after_train = time.time()
        print("Epoch {}, Time {:.4f}, Train loss {}".format(e, after_train - before_train, train_loss))

        # Evaluation
        if e % arguments.eval_freq == 0:
            before_train = time.time()
            train_net.set_train(False)

            # Raw
            head_meanrank_raw = 0
            head_hits10_raw = 0
            tail_meanrank_raw = 0
            tail_hits10_raw = 0

            # Filter
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
            after_train = time.time()
            print("Eval Time {}".format(after_train - before_train))
            print("Raw:     MeanRank {:.3f}, Hits@10 {:.3f}".format((head_meanrank_raw + tail_meanrank_raw) / 2,
                                                                    (head_hits10_raw + tail_hits10_raw) / 2))
            head_meanrank_filter /= len(eval_dataloader)
            head_hits10_filter /= len(eval_dataloader)
            tail_meanrank_filter /= len(eval_dataloader)
            tail_hits10_filter /= len(eval_dataloader)
            print("Filter:  MeanRank {:.3f}, Hits@10 {:.3f}".format((head_meanrank_filter + tail_meanrank_filter) / 2,
                                                                    (head_hits10_filter + tail_hits10_filter) / 2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TransE')
    parser.add_argument('--data_path', type=str, help='Path to dataset')
    parser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    parser.add_argument('--embedding_dim', type=int, default=100, help='Embedding size')
    parser.add_argument('--margin_value', type=float, default=1.0, help='margin value')
    parser.add_argument('--score_func', type=str, default='L1', help='score function, L1 or L2')
    parser.add_argument('--batch_size', type=int, default=10000, help='size for a single batch')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
    parser.add_argument('--workers', type=int, default=24, help='use how many workers')
    parser.add_argument('--max_epoch', type=int, default=4000, help='maximum epoch')
    parser.add_argument('--eval_freq', type=int, default=100, help='evaluation frequency')
    parser.add_argument('--fuse', type=bool, default=True, help="whether to use graph mode")
    args = parser.parse_args()
    print(args)
    main(args)
