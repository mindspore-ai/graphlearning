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
"""dataset definition"""
import numpy as np


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
