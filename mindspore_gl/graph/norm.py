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
"""normalization"""
import mindspore as ms
from mindspore import ops
from mindspore_gl.graph import get_laplacian

def norm(edge_index, num_nodes, edge_weight=None, normalization='sym',
         lambda_max=None, batch=None):
    r"""
    graph laplacian normalization

    Args:
        edge_index (Tensor): Edge index. The shape is :math:`(2, N\_e)`
            where :math:`N\_e` is the number of edges.
        num_nodes (int): Number of nodes.
        edge_weight (Tensor): Edge weights. The shape is :math:`(N\_e)`
            where :math:`N\_e` is the number of edges. Default: dense.
        normalization (str): Normalization method. Default: 'sym'.

            1. `None`: No normalization
               :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`
            2. `'sym'`: Symmetric normalization
               :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
               \mathbf{D}^{-1/2}`
            3. `'rw'`: Random-walk normalization
               :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`
        lambda_max (int, float): Lambda value of graph. Default: dense.
        batch (Tensor): Batch vector. Default: dense.

    Returns:
        - **edge_index** (Tensor) - normalized edge_index.
        - **edge_weight** (Tensor) - normalized edge_weight

    Raises:
        ValueError: if `normalization` not is None or 'sym' or 'rw'.

    Examples:
        >>> import mindspore as ms
        >>> from mindspore_gl.graph import norm
        >>> edge_index = [[1, 1, 2, 2], [0, 2, 0, 1]]
        >>> edge_index = ms.Tensor(edge_index, ms.int32)
        >>> num_nodes = 3
        >>> edge_index, edge_weight = norm(edge_index, num_nodes)
        >>> print(edge_index)
        [[1 1 2 2 0 1 2]
         [0 2 0 1 0 1 2]]
        >>> print(edge_weight)
        [-0.        -0.4999999 -0.        -0.4999999  1.         1.
          1.       ]
    """
    assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

    edge_index, edge_weight = get_laplacian(edge_index, num_nodes, edge_weight,
                                            normalization)

    if lambda_max is None:
        lambda_max = 2.0 * edge_weight.max()
    elif not isinstance(lambda_max, ms.Tensor):
        lambda_max = ms.tensor(lambda_max, ms.float32)
    assert lambda_max is not None

    if batch is not None and lambda_max.size > 1:
        lambda_max = lambda_max[batch[edge_index[0]]]

    edge_weight = (2.0 * edge_weight) / lambda_max

    mask = ops.isfinite(edge_weight)
    mask = ops.logical_not(mask)
    edge_weight = ops.MaskedFill()(edge_weight, mask, 0.0)

    return edge_index, edge_weight
