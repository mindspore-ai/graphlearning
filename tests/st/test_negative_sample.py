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
"""test sample"""
import numpy as np
from mindspore_gl import negative_sample


def test_negative_sample():
    """
    Feature:Test that the negative sample output shape is correct,
    and test negative samples for presence in positive samples

    Description:
    initialize edge
    positive = [[1, 2], [2, 3]]
    neg_len = 4

    Expectation:
    Negative sample edge is not inside positive sample edge,
    neg_len = neg.shape[0]
    """
    def ismember(a, b, tol=5):
        '''
        Returns whether a is in b.
        '''
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)
    positive = [[1, 2], [2, 3]]
    neg_len = 4
    neg = negative_sample(positive, 4, neg_len)

    assert ~ismember(neg, np.array(positive))
    assert neg_len == neg.shape[0]
