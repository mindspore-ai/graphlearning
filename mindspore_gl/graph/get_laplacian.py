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
"""laplacian normalization"""
from mindspore_gl.graph.self_loop import add_self_loop
import mindspore as ms
from mindspore import ops


def get_laplacian(edge_index, num_nodes, edge_weight=None, normalization='sym'):
    r"""
    Get laplacian matrix.

    Args:
        edge_index (Tensor): Edge index. The shape is :math:`(2, N\_e)`
            where :math:`N\_e` is the number of edges.
        num_nodes (int): Number of nodes.
        edge_weight (Tensor): Edge weights. The shape is :math:`(N\_e)`
            where :math:`N\_e` is the number of edges. Default: None.
        normalization (str): Normalization method. Default: sym.

            1. :obj:`None`: No normalization
               :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`
            2. :obj:`"sym"`: Symmetric normalization
               :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
               \mathbf{D}^{-1/2}`
            3. :obj:`"rw"`: Random-walk normalization
               :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

    Returns:
        - **edge_index** (Tensor) - normalized edge_index.
        - **edge_weight** (Tensor) - normalized edge_weight.

    Raises:
        ValueError: if `normalization` not is None or sym or rw.

    Examples:
        >>> import mindspore as ms
        >>> from mindspore_gl.graph import get_laplacian
        >>> edge_index = [[1, 1, 2, 2], [0, 2, 0, 1]]
        >>> edge_index = ms.Tensor(edge_index, ms.int32)
        >>> num_nodes = 3
        >>> edge_weight = ms.Tensor([1, 2, 1, 2], ms.float32)
        >>> edge_index, edge_weight = get_laplacian(edge_index, num_nodes, edge_weight, 'sym')
        >>> print(edge_index)
        [[1 1 2 2 0 1 2]
         [0 2 0 1 0 1 2]]
        >>> print(edge_weight)
        [-0.        -0.6666666 -0.        -0.6666666  1.         1.
          1.       ]
    """

    if normalization not in [None, 'sym', 'rw']:
        raise TypeError("Invalid normalization, normalization must be 'sym', 'rm' or None")

    if edge_weight is None:
        edge_weight = ms.ops.Ones()(edge_index.shape[1], ms.float32)
    row, col = edge_index[0], edge_index[1]

    out = ops.Zeros()(num_nodes, ms.float32)
    index = ops.ExpandDims()(row, -1)
    deg = ops.TensorScatterAdd()(out, index, edge_weight)
    fill_values = ms.ops.Ones()(num_nodes, ms.float32)
    if normalization is None:
        # L = D - A.
        edge_index, edge_weight = add_self_loop(edge_index, edge_weight, num_nodes, fill_values, 'coo')
        edge_weight = ops.Concat()((-edge_weight, deg))
    elif normalization == 'sym':
        # Compute A_norm = -D^{-1/2} A D^{-1/2}.
        deg_inv_sqrt = ops.Pow()(deg, -0.5)
        mask = ops.isfinite(deg_inv_sqrt)
        mask = ops.logical_not(mask)
        deg_inv_sqrt = ops.MaskedFill()(deg_inv_sqrt, mask, 0.0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        # # L = I - A_norm.
        edge_index, edge_weight = add_self_loop(edge_index, -edge_weight, num_nodes, fill_values, 'coo')
    else:
        # Compute A_norm = -D^{-1} A.
        deg_inv = 1.0 / deg
        mask = ops.isfinite(deg_inv)
        mask = ops.logical_not(mask)
        deg_inv = ops.MaskedFill()(deg_inv, mask, 0.0)
        edge_weight = deg_inv[row] * edge_weight
        # L = I - A_norm.
        edge_index, edge_weight = add_self_loop(edge_index, -edge_weight, num_nodes, fill_values, 'coo')
    return edge_index, edge_weight
