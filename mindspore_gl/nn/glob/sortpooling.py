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
"""Sort Pooling Layer."""
import mindspore as ms

from mindspore_gl import BatchedGraph
from .. import GNNCell


class SortPooling(GNNCell):
    r"""
    Apply sort pooling to the nodes in the graph.

    From the paper `End-to-End Deep Learning Architecture for Graph Classification <https://www.cse.wustl.edu/~ychen/public/DGCNN.pdf>`_.
    The sorting pool first sorts the node features in ascending order along the feature dimension,
    and then selects the ranking features of top-k nodes (sorted by the maximum value of each node).


    Args:
        k (int): Number of nodes to keep per graph.
    """

    def __init__(self, k):
        super().__init__()
        self.k = k

    # pylint: disable=arguments-differ
    def construct(self, x, g: BatchedGraph):
        """
        Construct function for SortPooling.

        Args:
            x (Tensor): input node features.
            g (BatchedGraph): input batched graph.

        Returns:
            Tensor, output representation for graphs.
        """
        x, _ = ms.ops.Sort()(x)
        ret, _ = g.batched_topk_nodes(x, self.k, -1)
        ret = ms.ops.Reshape()(ret, (-1, self.k * ms.ops.Shape()(x)[-1]))
        return ret
