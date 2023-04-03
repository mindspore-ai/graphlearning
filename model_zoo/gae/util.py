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
""" util """
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

def get_auc_score(adj_rec, edges_pos, edges_neg):
    """
    Output the link prediction matrix and positive and negative samples and process them
    return AUC and AP scores

    Args:
        adj_rec(array),Link prediction matrix, shape :math:`(node, node)`
        edges_pos(array):positive edge, shape :math:`(pos_len, 2)`
        edges_neg(array):negative edge, shape :math:`(neg_len, 2)`

    Returns:
        auc_score(float):AUC score
        ap_score(float):AP score
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    assert edges_pos.shape[0] == edges_neg.shape[0]

    preds = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[int(e[0]), int(e[1])]))


    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])

    auc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return auc_score, ap_score
