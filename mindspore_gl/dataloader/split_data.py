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
""" split_data """
import numpy as np
import scipy.sparse as sp

def split_data(x, val_ratio=0.05, test_ratio=0.1, graph_type='undirected'):
    """
    Cut the training set into training set, validation set and test set according to the proportion of user input,
    and perform graph reconstruction on the training set, and then return

    Args:
        x: Graph Structured Dataset
        val_ratio(float): Validation set proportion
        test_ratio(float): Test set proportion
        graph_type(str):The type of graph

    Returns:
        g (Graph), Graph of the training set
        train(array), Train set positive examples,shape:(train_len, 2)
        val(array), Validation set positive example,shape:(val_len, 2)
        test(array), Test set positive examples,shape:(test_len, 2)

    Examples:
        >>> from mindspore_gl.dataloader import split_data
        >>> from mindspore_gl.dataset import CoraV2
        >>> ds = CoraV2('data_path')
        >>> adj_coo, (train, val, test) = split_data(ds)
        >>> print(train.shape, val.shape, test.shape)
        (11684, 2) (263, 2) (527, 2)
    """
    col = x.adj_coo.col
    row = x.adj_coo.row

    # Construct an adjacency matrix
    adj = []
    for i in range(len(col)):
        idx = []
        idx.append(col[i])
        idx.append(row[i])
        adj.append(idx)

    # Take the upper triangular matrix
    adj_c = [i for i in adj if i[0] != i[1]]
    if graph_type == 'undirected':
        adj_cc = []
        for i in adj_c:
            if [i[1], i[0]] not in adj_cc:
                adj_cc.append(i)
    else:
        adj_cc = adj_c


    # Shuffle the subscript order, split the validation set and the test set
    np.random.shuffle(adj_cc)
    s = len(adj_cc)
    val_l = int(s*val_ratio)
    test_l = int(s*test_ratio)
    idx = np.random.randint(val_l+test_l, s-val_l-test_l)
    val = adj_cc[idx:idx+val_l]
    test = adj_cc[idx+val_l:idx+val_l+test_l]

    # Remove the validation and test sets from the training set
    for i in val+test:
        if i in adj:
            adj.remove([i[1], i[0]])
            adj.remove([i[0], i[1]])
    train = adj
    adj, val, test, train = np.array(adj), np.array(val), np.array(test), np.array(train)

    # Refactored graph
    data = np.ones(train.shape[0])
    adj_train = sp.csr_matrix((data, (train[:, 0], train[:, 1])), shape=x.adj_coo.shape).tocoo(copy=False)

    return adj_train, (train, val, test)
