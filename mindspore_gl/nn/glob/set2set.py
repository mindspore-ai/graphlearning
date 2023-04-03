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
"""Set2Set Layer."""
import mindspore as ms
import mindspore.ops as ops
from mindspore_gl import BatchedGraph
from .. import GNNCell


class Set2Set(GNNCell):
    r"""
    Sequence to sequence for sets.
    From the paper `Order Matters: Sequence to sequence for sets <https://arxiv.org/abs/1511.06391>`_ .

    For each subgraph in the batched graph, compute:

    .. math::
        q_t = \mathrm{LSTM} (q^*_{t-1}) \\

        \alpha_{i,t} = \mathrm{softmax}(x_i \cdot q_t) \\

        r_t = \sum_{i=1}^N \alpha_{i,t} x_i\\

        q^*_t = q_t \Vert r_t

    Args:
        input_size (int): dim for input node features.
        num_iters (int): number of iterations.
        num_layers (int): number of layers.

    Inputs:
        - **x** (Tensor) - The input node features to be updated. The shape is :math:`(N, D)`
          where :math:`N` is the number of nodes, and :math:`D` is the feature size of nodes.
        - **g** (BatchedGraph) - The input graph.

    Outputs:
        - **x** (Tensor) - The output representation for graphs. The shape is :math:`(2, D_{out})`
          where :math:`D_{out}` is the double feature size of nodes

    Raises:
        TypeError: If `input_size` or `num_iters` or `num_layers` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore_gl.nn import Set2Set
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
        >>> net = Set2Set(4, 3, 2)
        >>> ret = net(node_feat, *batched_graph_field.get_batched_graph())
        >>> print(ret.shape)
        (2, 8)
    """

    def __init__(self, input_size, num_iters, num_layers):
        super().__init__()
        assert isinstance(input_size, int) and input_size > 0, "input_size must be positive int"
        assert isinstance(num_iters, int) and num_iters > 0, "num_iters must be positive int"
        assert isinstance(num_layers, int) and num_layers > 0, "num_layers must be positive int"
        self.input_size = input_size
        self.num_iters = num_iters
        self.num_layers = num_layers
        self.output_size = input_size * 2
        self.lstm = ms.nn.LSTM(self.output_size, self.input_size, self.num_layers)

    # pylint: disable=arguments-differ
    def construct(self, x, g: BatchedGraph):
        """
        Construct function for Set2Set.
        """
        batch_size = ops.Shape()(g.graph_mask)[0]

        h = (ops.Zeros()((self.num_layers, batch_size, self.input_size), ms.float32),
             ops.Zeros()((self.num_layers, batch_size, self.input_size), ms.float32))

        q_star = ops.Zeros()((batch_size, self.output_size), ms.float32)

        for _ in range(self.num_iters):
            q, h = self.lstm(ops.ExpandDims()(q_star, 0), h)
            q = ops.Reshape()(q, (batch_size, self.input_size))
            e = x * g.broadcast_nodes(q)
            e_sum = ops.ReduceSum(True)(e, -1)
            alpha = g.softmax_nodes(e_sum)
            r = x * alpha
            readout = g.sum_nodes(r)
            q_star = ops.Concat(-1)((q, readout))

        return q_star
