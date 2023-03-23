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
"""Weight And Sum Layer"""
import mindspore as ms
from mindspore._checkparam import Validator
from mindspore_gl import BatchedGraph
from .. import GNNCell


class WeightAndSum(GNNCell):
    """
    Calculates importance weights for nodes and performs weighted sums.

    Args:
        in_feat_size (int): input feature size.

    Inputs:
        - **x** (Tensor) - The input node features to be updated. The shape is :math:`(N, D)`
          where :math:`N` is the number of nodes, and :math:`D` is the feature size of nodes.
        - **g** (BatchedGraph) - The input graph.

    Outputs:
        - **x** (Tensor) - The output representation for graphs. The shape is :math:`(2, D_{out})`
          where :math:`D_{out}` is the feature size of nodes

    Raises:
        TypeError: If `in_feat_size` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore_gl.nn import WeightAndSum
        >>> from mindspore_gl import BatchedGraphField
        >>> n_nodes = 7
        >>> n_edges = 8
        >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
        >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
        >>> ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
        >>> edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
        >>> graph_mask = ms.Tensor([1, 1], ms.int32)
        >>> batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx,
        ...                                         edge_subgraph_idx, graph_mask)
        >>> node_feat = np.random.random((n_nodes, 4))
        >>> node_feat = ms.Tensor(node_feat, ms.float32)
        >>> net = WeightAndSum(4)
        >>> ret = net(node_feat, *batched_graph_field.get_batched_graph())
        >>> print(ret.shape)
        (2, 4)
    """

    def __init__(self, in_feat_size):
        super().__init__()
        self.in_feat_size = Validator.check_positive_int(in_feat_size, "in_feat_size", self.cls_name)
        self.atom_weighting = ms.nn.SequentialCell(
            ms.nn.Dense(in_feat_size, 1),
            ms.nn.Sigmoid()
        )

    # pylint: disable=arguments-differ
    def construct(self, x, g: BatchedGraph):
        """
        Construct function for WeightAndSum.
        """
        w = self.atom_weighting(x)
        return g.sum_nodes(x * w)
