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
from mindspore_gl import BatchedGraph
from .. import GNNCell


class GlobalAttentionPooling(GNNCell):
    r"""
    Apply global attention pooling to the nodes in the graph.
    From the paper `Gated Graph Sequence Neural Networks <https://arxiv.org/pdf/1511.05493.pdf>`_.

    .. math::
        r^{(i)} = \sum_{k=1}^{N_i}\mathrm{softmax}\left(f_{gate}
            \left(x^{(i)}_k\right)\right) f_{feat}\left(x^{(i)}_k\right)

    Args:
        gate_nn (Cell): The neural network for computing attention score for each feature.
        feat_nn (Cell): The neural network applied to each feature
            before combining each feature with an attention score.
    """

    def __init__(self, gate_nn, feat_nn=None):
        super().__init__()
        self.gate_nn = gate_nn
        self.feat_nn = feat_nn

    # pylint: disable=arguments-differ
    def construct(self, x, g: BatchedGraph):
        """
        Construct function for GlobalAttentionPooling.

        Args:
            x (Tensor): input node features.
            g (BatchedGraph): input batched graph.

        Returns:
            Tensor, output representation for graphs.
        """
        gate = self.gate_nn(x)
        # assert ms.ops.Shape()(x)[-1] == 1, "The output of gate_nn should have 1 at its last axis."
        x = self.feat_nn(x) if self.feat_nn else x
        gate = g.softmax_nodes(gate)
        x = x * gate
        readout = g.sum_nodes(x)
        return readout
