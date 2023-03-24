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
"""GMMConv Layer"""
import math
import mindspore as ms
from mindspore.common.initializer import XavierUniform
from mindspore._checkparam import Validator
from mindspore_gl import Graph
from .. import GNNCell


class GMMConv(GNNCell):
    r"""
    Gaussian mixture model convolutional layer.
    From the paper `Geometric Deep Learning on Graphs and Manifolds using Mixture Model CNNs <http://openaccess.thecvf.com/content_cvpr_2017/papers/Monti_Geometric_Deep_Learning_CVPR_2017_paper.pdf>`_ .

    .. math::
        u_{ij} = f(x_i, x_j), x_j \in \mathcal{N}(i) \\
        w_k(u) = \exp\left(-\frac{1}{2}(u-\mu_k)^T \Sigma_k^{-1} (u - \mu_k)\right) \\
        h_i^{l+1} = \mathrm{aggregate}\left(\left\{\frac{1}{K}
         \sum_{k}^{K} w_k(u_{ij}), \forall j\in \mathcal{N}(i)\right\}\right)

    where :math:`u` represents the pseudo coordinate between the vertex and one of its neighbors, computed using the
    function :math:`f`, where :math:`\Sigma_k^{-1}` and :math:`\mu_k` are the learnable parameters of the covariance
    matrix and the mean vector of the Gaussian kernel.

    Args:
        in_feat_size (int): Input node feature size.
        out_feat_size (int): Output node feature size.
        coord_dim (int): Dimension of pseudo-coordinates.
        n_kernels (int): Number of kernels.
        residual (bool, optional): Whether use residual. Default: False.
        bias (bool, optional): Whether use bias. Default: False.
        aggregator_type (str, optional): Type of aggregator, should be 'sum'. Default: 'sum'.

    Inputs:
        - **x** (Tensor) - The input node features. The shape is :math:`(N, D_{in})`
          where :math:`N` is the number of nodes,
          and :math:`D_{in}` should be equal to `in_feat_size` in `Args`.
        - **pseudo** (Tensor) - Pseudo coordinate tensor.
        - **g** (Graph) - The input graph.

    Outputs:
        - Tensor, output node features with shape of :math:`(N, D_{out})`, where :math:`(D_{out})` should be the same as
          `out_size` in `Args`.

    Raises:
        SyntaxError: when the aggregator type not equals to 'sum'.
        TypeError: If `in_feat_size` or `out_feat_size` or `coord_dim` or `n_kernels` is not an int.
        TypeError: If `bias` or `residual` is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore_gl.nn import GMMConv
        >>> from mindspore_gl import GraphField
        >>> n_nodes = 4
        >>> n_edges = 7
        >>> node_feat_size = 7
        >>> src_idx = ms.Tensor([0, 1, 1, 2, 2, 3, 3], ms.int32)
        >>> dst_idx = ms.Tensor([0, 0, 2, 1, 3, 0, 1], ms.int32)
        >>> ones = ms.ops.Ones()
        >>> node_feat = ones((n_nodes, node_feat_size), ms.float32)
        >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
        >>> meanconv = GMMConv(in_feat_size=node_feat_size, out_feat_size=2, coord_dim=3, n_kernels=2)
        >>> pseudo = ones((7, 3), ms.float32)
        >>> res = meanconv(node_feat, pseudo, *graph_field.get_graph())
        >>> print(res.shape)
        (4, 2)
    """

    def __init__(self,
                 in_feat_size: int,
                 out_feat_size: int,
                 coord_dim: int,
                 n_kernels: int,
                 residual=False,
                 bias=False,
                 aggregator_type="sum"):
        super().__init__()
        in_feat_size = Validator.check_positive_int(in_feat_size, "in_feat_size", self.cls_name)
        out_feat_size = Validator.check_positive_int(out_feat_size, "out_feat_size", self.cls_name)
        coord_dim = Validator.check_positive_int(coord_dim, "coord_dim", self.cls_name)
        n_kernels = Validator.check_positive_int(n_kernels, "n_kernels", self.cls_name)
        bias = Validator.check_bool(bias, "bias", self.cls_name)
        residual = Validator.check_bool(residual, "residual", self.cls_name)
        if aggregator_type != "sum":
            raise TypeError("Don't support aggregator type other than sum.")
        self.mu = ms.Parameter(
            ms.ops.normal((n_kernels, coord_dim), ms.Tensor([[0. for _ in range(coord_dim)]], ms.float32),
                          ms.Tensor([[0.1 for _ in range(coord_dim)]], ms.float32)))
        self.inv_sigma = ms.Parameter(ms.ops.Ones()((n_kernels, coord_dim), ms.float32))
        gain = math.sqrt(2)
        self.dense = ms.nn.Dense(in_feat_size, out_feat_size * n_kernels, has_bias=bias,
                                 weight_init=XavierUniform(gain))
        self.residual = None
        if residual:
            self.residual = ms.nn.Dense(in_feat_size, out_feat_size, has_bias=bias, weight_init=XavierUniform(gain))
        self.agg_type = aggregator_type
        self.n_kernels = n_kernels
        self.out_feat_size = out_feat_size
        self.coord_dim = coord_dim

    # pylint: disable=arguments-differ
    def construct(self, x, pseudo, g: Graph):
        """
        Construct function for GMMConv.
        """
        g.set_vertex_attr({"h": ms.ops.Reshape()(self.dense(x), (-1, self.n_kernels, self.out_feat_size))})
        gaussian = -0.5 * ((ms.ops.Reshape()(pseudo, (-1, 1, self.coord_dim)) - ms.ops.Reshape()(self.mu, (
            1, self.n_kernels, self.coord_dim))) ** 2)
        gaussian = gaussian * (ms.ops.Reshape()(self.inv_sigma, (1, self.n_kernels, self.coord_dim)) ** 2)
        gaussian = ms.ops.Exp()(ms.ops.ReduceSum(keep_dims=True)(gaussian, -1))
        g.set_edge_attr({"g": gaussian})
        for v in g.dst_vertex:
            e = [s.h * e.g for s, e in v.inedges]
            v.rt = g.sum(e)
            v.rt = ms.ops.ReduceSum()(v.rt, 1)
            if self.residual is not None:
                v.rt = v.rt + self.residual(v.h)
        return [v.rt for v in g.dst_vertex]
