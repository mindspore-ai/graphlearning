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
"""Average Pooling Layer"""
# pylint: disable=unused-import
import mindspore
from mindspore_gl import BatchedGraph
from .. import GNNCell


class AvgPooling(GNNCell):
    r"""
    Apply average pooling to the nodes in the batched graph.

    .. math::
        r^{(i)} = \frac{1}{N_i}\sum_{k=1}^{N_i} x^{(i)}_k
    """

    # pylint: disable=arguments-differ
    def construct(self, x, g: BatchedGraph):
        """
        Construct function for AvgPooling.

        Args:
            x (Tensor): input node features.
            g (BatchedGraph): input batched graph.

        Returns:
            Tensor, output representation for graphs.
        """
        return g.avg_nodes(x)
