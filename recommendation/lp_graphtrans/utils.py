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
import mindspore.ops as ops
import mindspore as ms


def pad_batch_ms(h_node, batch, max_input_len):
    """
    path_batch_ms. For batch nodes, need to mask some node.

    Args:
        h_node(Tensor): node embedding
        batch(Tensor): node graph
        max_input_len(int): transformer sequence
        get_mask(bool): mask node

    Returns: padded_h_node, src_padding_mask

    """

    num_batch = batch[-1] + 1
    num_nodes = []
    masks = []
    for i in range(num_batch):
        mask = ops.equal(batch, i)
        mask = [idx for idx, i in enumerate(mask) if i]
        masks.append(ms.Tensor(mask))
        num_nodes.append(len(mask))

    max_num_nodes = min(max(num_nodes), max_input_len)
    zeros_fn = ops.Zeros()
    padded_h_node = \
        zeros_fn((int(max_num_nodes),
                  int(num_batch), h_node.shape[-1]), ms.int32)
    src_padding_mask = zeros_fn((int(num_batch), int(max_num_nodes)), ms.int32)

    for i, mask in enumerate(masks):
        num_node = int(num_nodes[i])
        if num_node > max_num_nodes:
            num_node = int(max_num_nodes)
        h_node_tmp = h_node[mask]
        padded_h_node[-num_node:, i] = h_node_tmp[-num_node:]

        src_padding_mask[i, : max_num_nodes - num_node] = 1  # [b, s]
    src_padding_mask = src_padding_mask.astype(ms.bool_)

    return padded_h_node, src_padding_mask


if __name__ == '__main__':
    x = ms.Tensor(np.random.random((7, 4)), dtype=ms.float32)
    batch_demo = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)

    res = pad_batch_ms(x, batch_demo, 5)
    print(res[0])
