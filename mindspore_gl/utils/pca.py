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
""" pca """
import math
import numpy as np

def large_mat_mul(input_a, input_b, batch=32):
    """
    Large Matrix Slicing Operations.
    """
    m, _ = input_a.shape
    block_m = math.floor(m/batch)
    out = []
    for i in range(batch):
        start = i*block_m
        end = (i+1)*block_m
        new_a = input_a[start:end]
        out_i = np.matmul(new_a, input_b)
        out.append(out_i)
    out = np.concatenate(out, axis=0)
    remain_a = input_a[batch*block_m:m]
    remain_o = np.matmul(remain_a, input_b)
    output = np.concatenate((out, remain_o), axis=0)
    return output

def mat_mul(input_a, input_b):
    """
    Refactored matmul operation.
    """
    m, _ = input_a.shape
    if m > 100000:
        out = large_mat_mul(input_a, input_b)
    else:
        out = np.matmul(input_a, input_b)
    return out

def get_approximate_basis(matrix: np.ndarray, k: int = 6, niter: int = 2):
    """
    Return tensor Q with k orthonormal columns \
    such that 'Q Q^H matrix` approximates `matrix`.
    """
    niter = 2 if niter is None else niter
    _, n = matrix.shape[-2:]
    r = np.random.randn(n, k)
    matrix_t = matrix.T

    q, _ = np.linalg.qr(mat_mul(matrix, r))
    for _ in range(niter):
        q = np.linalg.qr(mat_mul(matrix_t, q))[0]
        q = np.linalg.qr(mat_mul(matrix, q))[0]
    return q

def pca(matrix: np.ndarray, k: int = None, niter: int = 2, norm: bool = False):
    r"""
    Perform a linear principal component analysis (PCA) on the matrix,
    and will return the first k dimensionality-reduced features.

    Args:
        matrix(ndarray): Input features, shape is :math:`(B, F)`.
        k(int): target dimension for dimensionality reduction. Default: None.
        niter(int): the number of subspace iterations to conduct
            and it must be a nonnegative integer. Default: 2.
        norm(bool): Whether the output is normalized. Default: False.

    Return:
        ndarray, Features after dimensionality reduction

    Raises:
        TypeError: If 'k' or 'niter' is not a positive int.
        TypeError: If 'matrix' is not a ndarry.
        TypeError: If 'norm' is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore_gl.utils import pca
        >>> X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        >>> data = pca(X, 1)
        >>> print(data)
        [[ 0.33702252]
        [ 2.22871406]
        [ 3.6021826 ]
        [-1.37346854]
        [-2.22871406]
        [-3.6021826 ]]
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("The matrix type is {},\
                        but it should be Tensor.".format(type(matrix)))
    if not isinstance(k, int) or k <= 0:
        raise TypeError("The k type is {},\
                        but it should be positive int.".format(type(k)))
    if not isinstance(niter, int) or niter <= 0:
        raise TypeError("The niter type is {},\
                        but it should be positive int.".format(type(niter)))
    if not isinstance(norm, bool):
        raise TypeError("The norm type is {},\
                        but it should be bool.".format(type(norm)))
    m, n = matrix.shape[-2:]
    if k is None:
        k = min(6, m, n)

    c = np.mean(matrix, axis=-2)
    norm_matrix = matrix - c

    q = get_approximate_basis(norm_matrix.T, k, niter)
    q_c = q.conjugate()
    b_t = mat_mul(norm_matrix, q_c)
    _, _, v = np.linalg.svd(b_t, full_matrices=False)
    v_c = v.conj().transpose(-2, -1)
    v_c = mat_mul(q, v_c)

    if not norm:
        matrix = mat_mul(matrix, v_c[:,])
    else:
        matrix = mat_mul(norm_matrix, v_c[:,])

    return matrix
