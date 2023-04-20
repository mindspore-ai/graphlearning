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
"""Global Attention Pooling Layer"""
# pylint: disable=unused-import
import mindspore
from mindspore import nn
from mindspore_gl import BatchedGraph
from .. import GNNCell


class GlobalAttentionPooling(GNNCell):
    r"""
    Apply global attention pooling to the nodes in the graph.
    From the paper `Gated Graph Sequence Neural Networks <https://arxiv.org/pdf/1511.05493.pdf>`_ .

    .. math::
        r^{(i)} = \sum_{k=1}^{N_i}\mathrm{softmax}\left(f_{gate}
            \left(x^{(i)}_k\right)\right) f_{feat}\left(x^{(i)}_k\right)

    Args:
        gate_nn (Cell): The neural network for computing attention score for each feature.
        feat_nn (Cell, optional): The neural network applied to each feature
            before combining each feature with an attention score. Default: ``None``.

    Inputs:
        - **x** (Tensor) - The input node features to be updated. The shape is :math:`(N, D)`
          where :math:`N` is the number of nodes, and :math:`D` is the feature size of nodes.
        - **g** (BatchedGraph) - The input graph.

    Outputs:
        - **x** (Tensor) - The output representation for graphs. The shape is :math:`(2, D_{out})`
          where :math:`D_{out}` is the feature size of nodes.

    Raises:
        TypeError: if `gate_nn` type or `feat_nn` type is not `mindspore.nn.Cell`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore_gl.nn import GlobalAttentionPooling
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
        >>> gate_nn = ms.nn.Dense(4, 1)
        >>> net = GlobalAttentionPooling(gate_nn)
        >>> ret = net(node_feat, *batched_graph_field.get_batched_graph())
        >>> print(ret.shape)
        (2, 4)
    """

    def __init__(self, gate_nn, feat_nn=None):
        super().__init__()
        if gate_nn:
            if not isinstance(gate_nn, nn.Cell):
                raise TypeError("gate_nn type should be mindspore.nn.Cell")
        if feat_nn:
            if not isinstance(feat_nn, nn.Cell):
                raise TypeError("feat_nn type should be mindspore.nn.Cell")
        self.gate_nn = gate_nn
        self.feat_nn = feat_nn

    # pylint: disable=arguments-differ
    def construct(self, x, g: BatchedGraph):
        """
        Construct function for GlobalAttentionPooling.
        """
        gate = self.gate_nn(x)
        # assert ms.ops.Shape()(x)[-1] == 1, "The output of gate_nn should have 1 at its last axis."
        x = self.feat_nn(x) if self.feat_nn else x
        gate = g.softmax_nodes(gate)
        x = x * gate
        readout = g.sum_nodes(x)
        return readout
