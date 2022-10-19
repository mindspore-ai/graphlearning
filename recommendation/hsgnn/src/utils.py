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
"""utils"""
import numpy as np


def take_rest(x, y):
    """get the rest of the index"""
    x.sort()
    y.sort()
    res = []
    j, jmax = 0, len(y)
    for i, _ in enumerate(x):
        flag = False
        while j < jmax and y[j] <= x[i]:
            if y[j] == x[i]:
                flag = True
            j += 1
        if not flag:
            res.append(x[i])
    return res


def random_splits(data, num_classes, percls_trn, val_lb, seed=42):
    """Random train-test splits"""
    index = [i for i in range(0, data.y.shape[0])]
    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y == c)[0]
        if len(class_idx) < percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(
                rnd_state.choice(class_idx, percls_trn, replace=False))

    if len(index) < 10000:
        rest_index = [i for i in index if i not in train_idx]
        val_idx = rnd_state.choice(rest_index, val_lb, replace=False)
        test_idx = [i for i in rest_index if i not in val_idx]
    else:
        rest_index = take_rest(index, train_idx)
        val_idx = rnd_state.choice(rest_index, val_lb, replace=False)
        test_idx = take_rest(rest_index, val_idx)

    data.train_mask = train_idx
    data.val_mask = val_idx
    data.test_mask = test_idx

    return data
