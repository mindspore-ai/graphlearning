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

from mindspore_gl import BatchedGraph
from .. import GNNCell


class WeightAndSum(GNNCell):
    """
    Calculates importance weights for nodes and performs weighted sums.

    Args:
        in_feat_size (int): input feature size.
    """

    def __init__(self, in_feat_size):
        super().__init__()
        self.in_feat_size = in_feat_size
        self.atom_weighting = ms.nn.SequentialCell(
            ms.nn.Dense(in_feat_size, 1),
            ms.nn.Sigmoid()
        )

    # pylint: disable=arguments-differ
    def construct(self, x, g: BatchedGraph):
        """
        Construct function for WeightAndSum.

        Args:
            x (Tensor): input node features.
            g (BatchedGraph): input batched graph.

        Returns:
            Tensor, output representation for graphs.
        """
        w = self.atom_weighting(x)
        return g.sum_nodes(x * w)
