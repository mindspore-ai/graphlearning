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
    From the paper `Order Matters: Sequence to sequence for sets <https://arxiv.org/abs/1511.06391>`_.

    For each subgraph in the batched graph, compute:

    .. math::
        q_t = \mathrm{LSTM} (q^*_{t-1}) \\

        \alpha_{i,t} = \mathrm{softmax}(x_i \cdot q_t) \\

        r_t = \sum_{i=1}^N \alpha_{i,t} x_i\\

        q^*_t = q_t \Vert r_t

    Args:
        input_size (int): dim for input node features.
        num_iters (int): number of iters.
        num_layers (int): number of layers.
    """

    def __init__(self, input_size, num_iters, num_layers):
        super().__init__()
        self.input_size = input_size
        self.output_size = input_size * 2
        self.num_iters = num_iters
        self.num_layers = num_layers
        self.lstm = ms.nn.LSTM(self.output_size, self.input_size, self.num_layers)

    # pylint: disable=arguments-differ
    def construct(self, x, g: BatchedGraph):
        """
        Construct function for Set2Set.

        Args:
            x (Tensor): input node features.
            g (BatchedGraph): input batched graph.

        Returns:
            Tensor, output representation for graphs.
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
