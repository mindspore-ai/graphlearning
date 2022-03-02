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
# ==============================================================================
"""
User-defined API for MindRecord GNN writer.
"""
import os.path as osp
import numpy as np
import scipy.sparse as sp
from mindspore_gl.temp import MindRecordDatatype, DEFAULT_DATA_SHAPE

data_dir = '/home/yuanxl/dataloader'

# profile:  (feature_names, feature_data_types, feature_shapes)
node_profile = (
    ["feat", "label"],
    [MindRecordDatatype.FLOAT32, MindRecordDatatype.INT32],
    [DEFAULT_DATA_SHAPE, DEFAULT_DATA_SHAPE]
)
edge_profile = ([], [], [])


def _normalize_cora_features(features):
    row_sum = np.array(features.sum(1))
    r_inv = np.power(row_sum * 1.0, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def _parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def yield_nodes(task_id=0):
    """
    Generate node dataloader
    Yields:
        dataloader (dict): dataloader row which is dict.
    """
    print("Node task is {}".format(task_id))

    # load graph dataloader
    data_file = osp.join(data_dir, "reddit_with_mask.npz")
    with np.load(data_file) as npz_file:
        node_feats = npz_file["feat"]
        node_labels = npz_file["label"]

        line_count = 0
        for i, label in enumerate(node_labels):
            node = {'id': i, 'type': 0, 'feature_1': node_feats[i].tolist(),
                    'feature_2': label}
            line_count += 1
            yield node
    print('Processed {} lines for nodes.'.format(line_count))


def yield_edges(task_id=0):
    """
    Generate edge dataloader
    Yields:
        dataloader (dict): dataloader row which is dict.
    """
    print("Edge task is {}".format(task_id))

    # load graph dataloader
    data_file = osp.join(data_dir, "reddit_with_mask.npz")
    with np.load(data_file) as npz_file:
        csr_indptr = npz_file["adj_csr_indptr"]
        csr_indices = npz_file["adj_csr_indices"]
        node_num = npz_file['n_nodes']
        line_count = 0
        for index in range(node_num):
            col_start = csr_indptr[index]
            col_end = csr_indptr[index + 1]
            for edge_id in range(col_start, col_end):
                edge = {'id': edge_id, 'src_id': index, 'dst_id': int(csr_indices[edge_id]), 'type': 0}
                line_count += 1
                yield edge
        print('Processed {} lines for edges.'.format(line_count))


def variants():
    """
    return variants
    """
    data_file = osp.join(data_dir, "reddit_with_mask.npz")
    ret = {}
    with np.load(data_file) as npz_file:
        ret['train_mask'] = npz_file["train_mask"].tolist()
        ret['test_mask'] = npz_file["test_mask"].tolist()
        ret['val_mask'] = npz_file["val_mask"].tolist()
        ret['n_classes'] = npz_file["n_classes"].tolist()
        ret['n_edges'] = npz_file["n_edges"].tolist()
        ret['n_nodes'] = npz_file["n_nodes"].tolist()
    return ret
