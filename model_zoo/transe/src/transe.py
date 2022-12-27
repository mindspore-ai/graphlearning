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
"""TransE"""
import math
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Uniform


class TransE(nn.Cell):
    """TransE"""
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
        """construct function"""
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
