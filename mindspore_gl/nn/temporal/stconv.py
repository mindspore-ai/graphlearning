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
"""STGCN layer"""
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from mindspore._checkparam import Validator
from mindspore_gl.nn.conv.chebconv import ChebConv
from mindspore_gl.nn import GNNCell
from mindspore_gl import Graph

class TemporalConv(ms.nn.Cell):
    """temporal convolution layer
    from the paper `A deep learning framework for traffic forecasting
    arXiv preprint arXiv:1709.04875, 2017. <https://arxiv.org/pdf/1709.04875.pdf>`_ .

    Args:
        in_channels (int): Input node feature size.
        out_channels (int): Output node feature size.
        kernel_size (int): Convolutional kernel size.

    Inputs:
        - **x** (Tensor) - The input node features. The shape is :math:`(B, T, N, (D_{in}))`
          where :math:`B` is the size of batch, :math:`T` is the number of input time steps,
          :math:`N` is the number of nodes,
          :math:`(D_{in})` should be equal to `in_channels` in `Args`.

    Outputs:
        - Tensor, output node features with shape of :math:`(B, D_{out}, N, T)`,
        where :math:`B` is the size of batch, :math:`(D_{out})` should be the same as
        `out_channels` in `Args`, :math:`N` is the number of nodes,
        :math:`T` is the number of input time steps.

    Raises:
        TypeError: If `in_channels` or `out_channels` or `kernel_size` is not an int.

    Examples:
        >>> import mindspore as ms
        >>> from mindspore_gl.nn.temporal import TemporalConv
        >>> import numpy as np
        >>> batch_size = 4
        >>> input_time_steps = 6
        >>> num_nodes = 2
        >>> in_channels = 2
        >>> out_channels = 1
        >>> temprol_conv = TemporalConv(in_channels, out_channels)
        >>> input = ms.Tensor(np.ones((batch_size, input_time_steps, num_nodes, in_channels)), ms.float32)
        >>> out = temprol_conv(input)
        >>> print(out.shape)
        (4, 4, 2, 1)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(TemporalConv, self).__init__()

        self.in_channels = Validator.check_positive_int(in_channels, "in_channels", self.cls_name)
        self.out_channels = Validator.check_positive_int(out_channels, "out_channels", self.cls_name)
        self.kernel_size = Validator.check_positive_int(kernel_size, "kernel_size", self.cls_name)

        self.conv_1 = ms.nn.Conv2d(in_channels, out_channels, (1, kernel_size), pad_mode='valid', has_bias=True)
        self.conv_2 = ms.nn.Conv2d(in_channels, out_channels, (1, kernel_size), pad_mode='valid', has_bias=True)
        self.conv_3 = ms.nn.Conv2d(in_channels, out_channels, (1, kernel_size), pad_mode='valid', has_bias=True)

        self.sigmoid = ms.nn.Sigmoid()
        self.relu = ms.nn.ReLU()

    def construct(self, x):
        """
        Construct function for temporal convolution layer.
        """
        x = ops.Transpose()(x, (0, 3, 2, 1))
        p = self.conv_1(x)
        q = self.sigmoid(self.conv_2(x))
        pq = p * q
        h = self.relu(pq + self.conv_3(x))
        h = ops.Transpose()(h, (0, 3, 2, 1))
        return h

class STConv(GNNCell):
    r"""
    Spatial-Temporal Graph Convolutional layer.
    From the paper `A deep learning framework for traffic forecasting
    arXiv preprint arXiv:1709.04875, 2017. <https://arxiv.org/pdf/1709.04875.pdf>`_ .
    The STGCN layer contains 2 temporal convolution layer and 1
    graph convolution layer (ChebyNet).

    Args:
        num_nodes (int): number of nodes.
        in_channels (int): Input node feature size.
        hidden_channels (int): hidden feature size.
        out_channels (int): Output node feature size.
        kernel_size (int, optional): Convolutional kernel size. Default: 3.
        k (int, optional): Chebyshev filter size. Default: 3.
        bias (bool, optional): Whether use bias. Default: True.

    Inputs:
        - **x** (Tensor) - The input node features. The shape is :math:`(B, T, N, (D_{in}))`
          where :math:`B` is the size of batch, :math:`T` is the number of input time steps,
          :math:`N` is the number of nodes,
          :math:`(D_{in})` should be equal to `in_channels` in `Args`.
        - **edge_weight** (Tensor) - Edge weights. The shape is :math:`(N\_e,)`
          where :math:`N\_e` is the number of edges.
        - **g** (Graph) - The input graph.

    Outputs:
        - Tensor, output node features with shape of :math:`(B, D_{out}, N, T)`,
          where :math:`B` is the size of batch, :math:`(D_{out})` should be the same as
          `out_channels` in `Args`, :math:`N` is the number of nodes,
          :math:`T` is the number of input time steps.

    Raises:
        TypeError: If `num_nodes` or `in_channels` or `out_channels` or `hidden_channels`
            or `kernel_size` or is `k` not an int.
        TypeError: If `bias` is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore_gl.nn.temporal import STConv
        >>> from mindspore_gl import GraphField
        >>> from mindspore_gl.graph import norm
        >>> n_nodes = 4
        >>> n_edges = 6
        >>> feat_size = 2
        >>> edge_attr = ms.Tensor([1, 1, 1, 1, 1, 1], ms.float32)
        >>> edge_index = ms.Tensor([[1, 1, 2, 2, 3, 3],
        >>>                         [0, 2, 1, 3, 0, 1]], ms.int32)
        >>> edge_index, edge_weight = norm(edge_index, n_nodes, edge_attr, 'sym')
        >>> edge_weight = ms.ops.Reshape()(edge_weight, ms.ops.Shape()(edge_weight) + (1,))
        >>> batch_size = 2
        >>> input_time_steps = 5
        >>> feat = ms.Tensor(np.ones((batch_size, input_time_steps, n_nodes, feat_size)), ms.float32)
        >>> graph_field = GraphField(edge_index[0], edge_index[1], n_nodes, n_edges)
        >>> stconv = STConv(num_nodes=n_nodes, in_channels=feat_size,
        >>>                 hidden_channels=3, out_channels=2,
        >>>                 kernel_size=2, k=2)
        >>> out = stconv(feat, edge_weight, *graph_field.get_graph())
        >>> print(out.shape)
        (2, 3, 4, 2)
    """
    def __init__(self,
                 num_nodes: int,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 k: int = 3,
                 bias: bool = True):
        super().__init__()
        self.num_nodes = Validator.check_positive_int(num_nodes, "num_nodes", self.cls_name)
        self.in_channels = Validator.check_positive_int(in_channels, "in_channels", self.cls_name)
        self.hidden_channels = Validator.check_positive_int(hidden_channels, "hidden_channels", self.cls_name)
        self.out_channels = Validator.check_positive_int(out_channels, "out_channels", self.cls_name)
        self.kernel_size = Validator.check_positive_int(kernel_size, "kernel_size", self.cls_name)
        self.k = Validator.check_positive_int(k, "k", self.cls_name)
        bias = Validator.check_bool(bias, "bias", self.cls_name)

        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.k = k
        self.bias = bias

        self.temporala_conv1 = TemporalConv(
            in_channels=self.in_channels,
            out_channels=self.hidden_channels,
            kernel_size=self.kernel_size
        )
        self.cheb_conv = ChebConv(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            k=self.k,
            bias=self.bias,
        )
        self.temporala_conv2 = TemporalConv(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size
        )
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.relu = nn.ReLU()

    def construct(self, x, edge_weight, g: Graph):
        """
        Construct function for STConv.
        """
        t0 = self.temporala_conv1(x)
        t = ops.ZerosLike()(t0)
        for b in range(t0.shape[0]):
            for s in range(t0.shape[1]):
                t[b][s] = self.cheb_conv(t0[b][s], edge_weight, g)
        t = self.relu(t0)
        t = self.temporala_conv2(t)
        t = ops.Transpose()(t, (0, 2, 1, 3))
        t = self.batch_norm(t)
        t = ops.Transpose()(t, (0, 2, 1, 3))
        return t
