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
import time
import random
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.context as context
import mindspore.dataset as ds

from mindspore_gl.dataloader import RandomBatchSampler, Dataset
from src.transe import TransE
from src.dataset import KnowLedgeGraphDataset


class TrainDataset(Dataset):
    """train dataset"""

    def __init__(self, kg: KnowLedgeGraphDataset, length):
        super().__init__()
        self.kg = kg
        self.length = length

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

    def __len__(self):
        return self.length


e_emb = None
r_emb = None


class EvalDataset(Dataset):
    """Eval Dataset"""

    def __init__(self, kg: KnowLedgeGraphDataset, score_func, length):
        super().__init__()
        self.kg = kg
        self.score_func = score_func
        self.length = length

    def __getitem__(self, idx):
        idx = idx[0]
        h, t, r = self.kg.triples[idx]
        global e_emb, r_emb
        return evaluate(e_emb, r_emb, h, t, r, self.score_func, self.kg)

    def __len__(self):
        return self.length


def get_rank(kg, head_position, tail_position, head, tail, relation):
    """Get rank"""
    head_r_raw = 0
    tail_r_raw = 0
    head_r_filter = 0
    tail_r_filter = 0
    for candidate in head_position:
        if candidate == head:
            break
        head_r_raw += 1
        if (candidate, tail, relation) not in kg.triple_pool:
            head_r_filter += 1
    for candidate in tail_position:
        if candidate == tail:
            break
        tail_r_raw += 1
        if (head, candidate, relation) not in kg.triple_pool:
            tail_r_filter += 1
    return head_r_raw, tail_r_raw, head_r_filter, tail_r_filter


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
        context.set_context(device_target="GPU", mode=context.GRAPH_MODE)

    kg = KnowLedgeGraphDataset(arguments.data_path)
    train_batch_sampler = RandomBatchSampler(kg.train_mask, batch_size=arguments.batch_size)
    train_dataset = TrainDataset(kg, len(list(train_batch_sampler)))
    train_dataloader = ds.GeneratorDataset(train_dataset, ['head_pos', 'tail_pos', 'relation_pos',
                                                           'head_neg', 'tail_neg', 'relation_neg'],
                                           sampler=train_batch_sampler, python_multiprocessing=True)
    eval_batch_sampler = RandomBatchSampler(kg.test_mask, batch_size=1)
    eval_dataset = EvalDataset(kg, arguments.score_func, len(list(eval_batch_sampler)))
    eval_dataloader = ds.GeneratorDataset(eval_dataset, ['head_r_raw', 'tail_r_raw', 'head_r_filter',
                                                         'tail_r_filter'],
                                          sampler=eval_batch_sampler, python_multiprocessing=True)

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
            head_mr_raw = 0
            head_h10_raw = 0
            tail_mr_raw = 0
            tail_h10_raw = 0

            # Filter
            head_mr_filter = 0
            head_h10_filter = 0
            tail_mr_filter = 0
            tail_h10_filter = 0

            global e_emb, r_emb
            e_emb = net.e_emb.embedding_table.asnumpy()
            r_emb = net.r_emb.embedding_table.asnumpy()
            for _, data in enumerate(eval_dataloader):
                head_r_raw, tail_r_raw, head_r_filter, tail_r_filter = data
                head_mr_raw += head_r_raw
                if head_r_raw < 10:
                    head_h10_raw += 1
                tail_mr_raw += tail_r_raw
                if tail_r_raw < 10:
                    tail_h10_raw += 1
                head_mr_filter += head_r_filter
                if head_r_filter < 10:
                    head_h10_filter += 1
                tail_mr_filter += tail_r_filter
                if tail_r_filter < 10:
                    tail_h10_filter += 1
            head_mr_raw /= len(eval_dataloader)
            head_h10_raw /= len(eval_dataloader)
            tail_mr_raw /= len(eval_dataloader)
            tail_h10_raw /= len(eval_dataloader)
            after_train = time.time()
            print("Eval Time {}".format(after_train - before_train))
            print("Raw:     MeanRank {:.3f}, Hits@10 {:.3f}".format((head_mr_raw + tail_mr_raw) / 2,
                                                                    (head_h10_raw + tail_h10_raw) / 2))
            head_mr_filter /= len(eval_dataloader)
            head_h10_filter /= len(eval_dataloader)
            tail_mr_filter /= len(eval_dataloader)
            tail_h10_filter /= len(eval_dataloader)
            print("Filter:  MeanRank {:.3f}, Hits@10 {:.3f}".format((head_mr_filter + tail_mr_filter) / 2,
                                                                    (head_h10_filter + tail_h10_filter) / 2))


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
