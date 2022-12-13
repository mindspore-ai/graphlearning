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
"""Deepwalk models"""
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Uniform

import mindspore_gl.sample_kernel as sample_kernel
from mindspore_gl.sampling import random_walk_unbias_on_homo
from mindspore_gl.dataloader import Dataset


class SkipGramModel(nn.Cell):
    """Skip Gram model"""
    def __init__(self,
                 num_nodes,
                 embed_size=16,
                 neg_num=5):
        super(SkipGramModel, self).__init__()

        self.num_nodes = num_nodes
        self.neg_num = neg_num

        self.s_emb = nn.Embedding(num_nodes, embed_size, embedding_table=Uniform(0.5/embed_size))
        self.d_emb = nn.Embedding(num_nodes, embed_size, embedding_table=Uniform(0))

        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.sum = ops.ReduceSum()
        self.matmul = nn.MatMul(transpose_x2=True)

    def construct(self, src, dsts, node_mask):
        """construct function"""
        # src [total pair count, 1]
        # dsts [total pair count, 1+neg]
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
    """Batch Randwalk"""
    def __init__(self, graph, walk_len, win_size, neg_num, batch_size):
        self.graph = graph
        # include head node itself
        self.walk_len = walk_len - 1
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
        node_mask = np.concatenate([np.ones((pair_count), np.int32), np.zeros((self.padded_size-pair_count), np.int32)])
        src = np.concatenate([src, np.ones((self.padded_size-pair_count, 1), np.int32)*self.fill_value])
        dsts = np.concatenate([dsts,
                               np.ones((self.padded_size-pair_count, 1 + self.neg_num), np.int32)*self.fill_value])
        node_mask = np.reshape(node_mask, [-1, 1, 1])

        return src, dsts, node_mask, pair_count


class DeepWalkDataset(Dataset):
    """Deepwalk dataset"""
    def __init__(self, nodes, batch_fn: BatchRandWalk, length: int, repeat=1):
        self.repeat = repeat
        self.data = nodes
        self.datalen = len(nodes)
        self.batch_fn = batch_fn
        self.length = length

    def __getitem__(self, batch_idxs):
        return self.batch_fn([self.data[idx % self.datalen] for idx in batch_idxs])

    def __len__(self):
        return self.length
