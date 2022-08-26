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
"""ASTGCN"""
from typing import Optional

import mindspore as ms
import mindspore.nn as nn
from mindspore.common.initializer import initializer
from mindspore._checkparam import Validator
from mindspore_gl import Graph
from .. import GNNCell


class SpatialAttention(nn.Cell):
    r"""
    An implementation of the Spatial Attention Module which adaptively capture the dynamic
    correlations between nodes in the spatial dimension.

    Args:
        in_channels (int): Number of input features.
        n_vertices (int): Number of vertices in the graph.
        num_of_timestamps (int): Number of time lags.
    """

    def __init__(
            self,
            in_channels: int,
            n_vertices: int,
            num_of_timestamps: int
        ):
        super(SpatialAttention, self).__init__()
        self.w1 = ms.Parameter(initializer('Uniform', (num_of_timestamps,), ms.float32), name="w1")
        self.w2 = ms.Parameter(initializer('XavierUniform', (in_channels, num_of_timestamps), ms.float32), name="w2")
        self.w3 = ms.Parameter(initializer('Uniform', (in_channels,), ms.float32), name="w3")
        self.bs = ms.Parameter(initializer('XavierUniform', (1, n_vertices, n_vertices), ms.float32),
                               name="bs")
        self.vs = ms.Parameter(initializer('XavierUniform', (n_vertices, n_vertices), ms.float32), name="vs")

        self.matmul = nn.MatMul()
        self.transpose = ms.ops.Transpose()
        self.sigmoid = ms.ops.Sigmoid()
        self.softmax = ms.ops.Softmax(axis=1)

    def construct(self, x):
        """
        Construct function for SpatialAttention.
        """
        x_w1 = self.matmul(x, self.w1)
        lhs = self.matmul(x_w1, self.w2)

        rhs = self.matmul(self.w3, x)
        rhs = self.transpose(rhs, (0, 2, 1))
        lrhs = self.matmul(lhs, rhs)
        sig = self.sigmoid(lrhs + self.bs)
        res = self.matmul(self.vs, sig)
        res = self.softmax(res)
        return res


class TemporalAttention(nn.Cell):
    r"""
    An implementation of the Temporal Attention Module. For details see the paper: `"Attention Based Spatial-Temporal
    Graph Convolutional Networks for Traffic Flow Forecasting."
    <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_.

    Args:
        in_channels (int): Number of input features.
        n_vertices (int): Number of vertices in the graph.
        num_of_timestamps (int): Number of time lags.
    """

    def __init__(self,
                 in_channels: int,
                 n_vertices: int,
                 num_of_timestamps: int,
                 ):
        super(TemporalAttention, self).__init__()
        self.u1 = ms.Parameter(initializer('Uniform', (n_vertices,), ms.float32), name="u1")
        self.u2 = ms.Parameter(initializer('XavierUniform', (in_channels, n_vertices), ms.float32), name="u2")
        self.u3 = ms.Parameter(initializer('Uniform', (in_channels,), ms.float32), name="u3")
        self.be = ms.Parameter(initializer('XavierUniform', (1, num_of_timestamps, num_of_timestamps), ms.float32),
                               name="be")
        self.ve = ms.Parameter(initializer('XavierUniform', (num_of_timestamps, num_of_timestamps), ms.float32),
                               name="ve")

        self.matmul = nn.MatMul()
        self.transpose = ms.ops.Transpose()
        self.sigmoid = ms.ops.Sigmoid()
        self.softmax = ms.ops.Softmax(axis=1)

    def construct(self, x):
        """
        Construct function for TemporalAttention.
        """
        x_t = self.transpose(x, (0, 3, 2, 1))
        x_u1 = self.matmul(x_t, self.u1)
        lhs = self.matmul(x_u1, self.u2)

        rhs = self.matmul(self.u3, x)

        lrhs = self.matmul(lhs, rhs)
        sig = self.sigmoid(lrhs + self.be)
        res = self.matmul(self.ve, sig)
        res = self.softmax(res)
        return res


class ChebConvAttention(GNNCell):
    r"""
    An implementation of the chebyshev spectral graph convolutional operator with attention.
    For details see the paper: `"Attention Based Spatial-Temporal Graph Convolutional Networks
    for Traffic Flow Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_.

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        normalization (str, optional): The normalization scheme for the graph Laplacian (default: "sym").
        bias (bool, optional): Whether the layer will learn an additive bias. (default: `True`)
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 k: int,
                 normalization: Optional[str] = None,
                 bias: bool = True,
                ):
        super(ChebConvAttention, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.k = k
        self.weight = ms.Parameter(initializer('XavierUniform', (k, in_channels, out_channels), ms.float32),
                                   name="weight")
        if bias:
            self.bias = ms.Parameter(initializer('Uniform', (out_channels), ms.float32), name="bias")
        else:
            self.bias = None

        self.permute = ms.ops.Transpose()
        self.matmul = nn.MatMul()
        self.eye = ms.ops.Eye()

    def construct(self, x, edge_weight, spatial_attention, g: Graph):
        """
        Construct function for ChebConvAttention.
        """

        num_nodes = x.shape[-2]
        col, row = g.src_idx, g.dst_idx
        att_norm = edge_weight * spatial_attention[row, col]
        identity = self.eye(num_nodes, num_nodes, ms.float32)
        tx_0 = identity * spatial_attention
        tx_0 = self.permute(tx_0, (1, 0))
        tx_0 = self.matmul(tx_0, x)
        tx_1 = tx_0
        out = self.matmul(tx_0, self.weight[0])

        if len(att_norm.shape) == 1:
            att_norm_reshape = att_norm.view(-1, 1)
        else:
            d1, d2 = att_norm.shape
            att_norm_reshape = att_norm.view(d1, d2, 1)

        if len(edge_weight.shape) == 1:
            edge_weight_reshape = edge_weight.view(-1, 1)
        else:
            d1, d2 = att_norm.shape
            edge_weight_reshape = edge_weight.view(d1, d2, 1)

        if self.k > 1:
            g.set_vertex_attr({"x": tx_0})
            for v in g.dst_vertex:
                feat = [u.x for u in v.innbs]
                v.x = g.sum(att_norm_reshape * feat)
            tx_1 = [v.x for v in g.dst_vertex]
            out += self.matmul(tx_1, self.weight[1])

        for k in range(2, self.k):
            g.set_vertex_attr({"x": tx_1})
            for v in g.dst_vertex:
                feat = [u.x for u in v.innbs]
                v.x = g.sum(edge_weight_reshape * feat)
            tx_2 = [v.x for v in g.dst_vertex]
            tx_2 = tx_2 * 2.0 - tx_0
            out += self.matmul(tx_2, self.weight[k])
            tx_0, tx_1 = tx_1, tx_2

        if self.bias is not None:
            out += self.bias

        return out


class ASTGCNBlock(GNNCell):
    r"""
    An implementation of the Attention Based Spatial-Temporal Graph Convolutional Block.
    For details see the paper: `"Attention Based Spatial-Temporal Graph Convolutional Networks
    for Traffic Flow Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_.

    Args:
        in_channels (int): Input node feature size.
        k (int): Order of Chebyshev polynomials.
        n_chev_filters (int): Number of Chebyshev filters.
        n_time_filters (int): Number of time filters.
        time_conv_strides (int): Time strides during temporal convolution.
        n_vertices (int): Number of vertices in the graph.
        num_of_timestamps (int): Number of time lags.
        normalization (str, optional): The normalization scheme for the graph Laplacian (default: "sym").
        bias (bool, optional): Whether the layer will learn an additive bias. (default: `True`)
    """

    def __init__(self,
                 in_channels: int,
                 k: int,
                 n_chev_filters: int,
                 n_time_filters: int,
                 time_conv_strides: int,
                 n_vertices: int,
                 num_of_timestamps: int,
                 normalization: Optional[str] = None,
                 bias: bool = True,
                 ) -> None:
        super(ASTGCNBlock, self).__init__()

        self.temporal_attention = TemporalAttention(in_channels, n_vertices, num_of_timestamps)
        self.spatial_attention = SpatialAttention(in_channels, n_vertices, num_of_timestamps)
        self.chebconv_attention = ChebConvAttention(in_channels, n_chev_filters, k, normalization, bias)
        self.time_conv = nn.Conv2d(in_channels=n_chev_filters,
                                   out_channels=n_chev_filters,
                                   kernel_size=(1, 3),
                                   stride=(1, time_conv_strides),
                                   pad_mode="pad",
                                   padding=(0, 0, 1, 1),
                                   has_bias=True)
        self.residual_conv = nn.Conv2d(in_channels=in_channels,
                                       out_channels=n_time_filters,
                                       kernel_size=(1, 1),
                                       stride=(1, time_conv_strides),
                                       has_bias=True)
        self.layer_norm = nn.LayerNorm([n_time_filters])
        self.normalization = normalization
        self.permute = ms.ops.Transpose()
        self.reshape = ms.ops.Reshape()
        self.matmul = nn.MatMul()
        self.unsqueeze = ms.ops.ExpandDims()
        self.relu = nn.ReLU()
        self.cat = ms.ops.Concat(axis=-1)

    # pylint: disable=arguments-differ
    def construct(self, x, edge_weight, g: Graph):
        """
        Construct function for ASTGCNBlock.
        """

        batch_size, n_vertices, num_of_features, num_of_timesteps = x.shape
        x_reshape = self.reshape(x, (batch_size, -1, num_of_timesteps))
        x_tilde = self.temporal_attention(x)
        x_tilde = self.matmul(x_reshape, x_tilde)
        x_tilde = self.reshape(x_tilde, (batch_size, n_vertices, num_of_features, num_of_timesteps))
        x_tilde = self.spatial_attention(x_tilde)

        x_hat = []
        cheb_conv_out = ms.ops.Zeros()(
            (batch_size,
             n_vertices,
             self.chebconv_attention.out_channels),
            ms.float32)
        if not isinstance(edge_weight, list):
            for t in range(num_of_timesteps):
                for b in range(batch_size):
                    cheb_conv_out[b] = self.chebconv_attention(
                        x[:, :, :, t][b],
                        edge_weight,
                        x_tilde[b],
                        g
                    )
                x_hat.append(self.unsqueeze(cheb_conv_out, -1))
        else:
            for t in range(num_of_timesteps):
                for b in range(batch_size):
                    cheb_conv_out[b] = self.chebconv_attention(
                        x[:, :, :, t][b],
                        edge_weight[t],
                        x_tilde[b],
                        g
                    )
                x_hat.append(self.unsqueeze(cheb_conv_out, -1))

        x_hat = self.relu(self.cat(x_hat))
        x_hat = self.permute(x_hat, (0, 2, 1, 3))
        x_hat = self.time_conv(x_hat)
        x = self.permute(x, (0, 2, 1, 3))
        x = self.residual_conv(x)
        x = self.relu(x + x_hat)
        x = self.permute(x, (0, 3, 2, 1))
        x = self.layer_norm(x)
        x = self.permute(x, (0, 2, 3, 1))
        return x


class ASTGCN(GNNCell):
    r"""
    Attention Based Spatial-Temporal Graph Convolutional Networks from the paper
    `Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic
    Flow Forecasting <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_.

    Args:
        n_blocks (int): Number of ASTGCN Blocks
        in_channels (int): Input node feature size.
        k (int): Order of Chebyshev polynomials.
        n_chev_filters (int): Number of Chebyshev filters.
        n_time_filters (int): Number of time filters.
        time_conv_strides (int): Time strides during temporal convolution.
        num_for_predict (int): Number of predictions to make in the future.
        len_input (int): Length of the input sequence.
        n_vertices (int): Number of vertices in the graph.
        normalization (str, optional): The normalization scheme for the graph Laplacian (default: "sym").
        bias (bool, optional): Whether the layer will learn an additive bias. (default: `True`)

    Inputs:
        - **x** (Tensor) - The input node features for T time periods. The shape is :math:`(B, N, F_{in}, T_{in})`
          where :math:`N` is the number of nodes,
        - **g** (Graph) - The input graph.

    Outputs:
        Tensor, output node features with shape of :math:`(B, N, T_{out})`.

    Raises:
        TypeError: If `n_blocks`, `in_channels`, `k`, `n_chev_filters`, `n_time_filters`, `time_conv_strides`,
                   `num_for_predict`, `len_input` or `n_vertices` is not a positive int.
        ValueError: If `normalization` is not 'sym'.

    Supported Platforms:
         ``GPU`` ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore_gl.graph import norm
        >>> from mindspore_gl.nn.temporal import ASTGCN
        >>> from mindspore_gl import GraphField
        >>> node_count = 5
        >>> num_for_predict = 4
        >>> len_input = 4
        >>> n_time_strides = 1
        >>> node_features = 2
        >>> nb_block = 2
        >>> k = 3
        >>> n_chev_filters = 8
        >>> n_time_filters = 8
        >>> batch_size = 2
        >>> normalization = "sym"
        >>> edge_index = np.array([[0, 0, 0, 0, 1, 1, 1, 2, 2, 3],
                                   [1, 4, 2, 3, 2, 3, 4, 3, 4, 4]])
        >>> model = ASTGCN(nb_block, node_features, k, n_chev_filters, n_time_filters,
                    n_time_strides, num_for_predict, len_input, node_count, normalization)
        >>> edge_index_norm, edge_weight_norm = norm(Tensor(edge_index, dtype=ms.int32), node_count)
        >>> graph = GraphField(edge_index_norm[1], edge_index_norm[0], node_count, len(edge_index_norm[0]))
        >>> x_seq = Tensor(np.ones([batch_size, node_count, node_features, len_input]), dtype=ms.float32)
        >>> output = model(x_seq, edge_weight_norm, *graph.get_graph())
        >>> print(output.shape)
        (2, 5, 4)
    """
    def __init__(self,
                 n_blocks: int,
                 in_channels: int,
                 k: int,
                 n_chev_filters: int,
                 n_time_filters: int,
                 time_conv_strides: int,
                 num_for_predict: int,
                 len_input: int,
                 n_vertices: int,
                 normalization: Optional[str] = None,
                 bias: bool = True,
                 ) -> None:
        super().__init__()
        self.n_blocks = Validator.check_positive_int(n_blocks, "n_blocks", self.cls_name)
        self.in_channels = Validator.check_positive_int(in_channels, "in_channels", self.cls_name)
        self.k = Validator.check_positive_int(k, "k", self.cls_name)
        self.n_chev_filters = Validator.check_positive_int(n_chev_filters, "n_chev_filters", self.cls_name)
        self.n_time_filters = Validator.check_positive_int(n_time_filters, "n_time_filters", self.cls_name)
        self.time_conv_strides = Validator.check_positive_int(time_conv_strides, "time_conv_strides", self.cls_name)
        self.num_for_predict = Validator.check_positive_int(num_for_predict, "num_for_predict", self.cls_name)
        self.len_input = Validator.check_positive_int(len_input, "len_input", self.cls_name)
        self.n_vertices = Validator.check_positive_int(n_vertices, "n_vertices", self.cls_name)
        if normalization not in ['sym']:
            raise ValueError(f"For '{self.cls_name}', the normalization: '{agg}' is unsupported.")
        self.normalization = normalization

        self.final_conv = nn.Conv2d(in_channels=int(len_input / time_conv_strides),
                                    out_channels=num_for_predict,
                                    kernel_size=(1, n_time_filters),
                                    pad_mode="valid",
                                    has_bias=True)
        self.permute = ms.ops.Transpose()

        blocks = nn.CellList()
        blocks.append(
            ASTGCNBlock(
                in_channels,
                k,
                n_chev_filters,
                n_time_filters,
                time_conv_strides,
                n_vertices,
                len_input,
                normalization,
                bias,
            )
        )
        blocks.extend(
            [
                ASTGCNBlock(
                    n_time_filters,
                    k,
                    n_chev_filters,
                    n_time_filters,
                    1,
                    n_vertices,
                    len_input // time_conv_strides,
                    normalization,
                    bias,
                )
                for _ in range(n_blocks - 1)
            ]
        )
        self.blocks = blocks

    # pylint: disable=arguments-differ
    def construct(self, x, edge_weight, g: Graph):
        """
        Construct function for ASTGCN.
        """

        for block in self.blocks:
            x = block(x, edge_weight, g)

        x = self.permute(x, (0, 3, 1, 2))
        x = self.final_conv(x)
        x = x[:, :, :, -1]
        x = self.permute(x, (0, 2, 1))
        return x
