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
import mindspore as ms

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

    q, _ = np.linalg.qr(np.matmul(matrix, r))
    for _ in range(niter):
        q = np.linalg.qr(np.matmul(matrix_t, q))[0]
        q = np.linalg.qr(np.matmul(matrix, q))[0]
    return q

def pca(matrix: ms.Tensor, k: int = None, niter: int = 2, norm: bool = False):
    r"""
    Perform a linear principal component analysis (PCA) on the matrix,
    and will return the first k dimensionality-reduced features.

    Args:
      matrix(Tensor): Input features, shape:(B, F)
      k(int): target dimension for dimensionality reduction
      niter(int): the number of subspace iterations to conduct \
      and it must be a nonnegative integer.
      norm(bool): Whether the output is normalized

    Return:
        Tensor, Features after dimensionality reduction

    Example:
      >>> import mindsprre as ms
      >>> from mindspore_gl import pca
      >>> X = ms.Tensor([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
      >>> data = pca(X, 1)
      >>> print(data)
          [[ 0.33702252]
          [ 2.22871406]
          [ 3.6021826 ]
          [-1.37346854]
          [-2.22871406]
          [-3.6021826 ]]
    """
    if not isinstance(matrix, ms.Tensor):
        raise TypeError("The matrix type is {},\
                        but it should be Tensor.".format(type(matrix)))
    if not isinstance(k, int):
        raise TypeError("The k type is {},\
                        but it should be int.".format(type(k)))
    m, n = matrix.shape[-2:]
    if k is None:
        k = min(6, m, n)
    matrix = matrix.asnumpy()

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

    return ms.Tensor(matrix)
