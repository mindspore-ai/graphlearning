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
"""APPNPConv Layer."""
import mindspore as ms
from mindspore import Tensor
from mindspore._checkparam import Validator


class GatherNet(ms.nn.Cell):
    """
    Redefine back-propagation to make use of sorted indices.
    """
    def __init__(self, indptr_backward, indices_backward):
        super().__init__()
        self.indptr_backward = indptr_backward
        self.indices_backward = indices_backward

    def construct(self, data, indices, axis):
        return ms.ops.gather(data, indices, axis)

    def bprop(self, data, indices, axis, out, dout):
        del indices, axis, out
        grad_csr = ms.CSRTensor(
            self.indptr_backward, self.indices_backward, dout, (data.shape[0], data.shape[0]) + data.shape[1:])
        grad_sum = grad_csr.sum(1).reshape(data.shape)
        return (grad_sum,)


class CSRReduceSumNet(ms.nn.Cell):
    """
    Redefine back-propagation to make use of sorted indices.
    """
    def __init__(self, indices_backward):
        super().__init__()
        self.indices_backward = indices_backward
        # pylint: disable=protected-access
        self.op = ms.ops.operations._csr_ops.CSRReduceSum()

    def construct(self, indptr, indices, values, shape, axis):
        return self.op(indptr, indices, values, shape, axis)

    def bprop(self, indptr, indices, values, shape, axis, out, dout):
        del values, shape, axis, out
        dout = dout.reshape((dout.shape[0],) + dout.shape[2:])
        grad_values = ms.ops.gather(dout, self.indices_backward, 0)
        return indptr, indices, grad_values, (), 0


class APPNPConv(ms.nn.Cell):
    r"""
    Approximate Personalization Propagation in Neural Prediction Layers.
    From the paper `Predict then Propagate: Graph Neural Networks meet Personalized PageRank <https://arxiv.org/pdf/1810.05997.pdf>`_.

    .. math::
        H^{0} = X \\
        H^{l+1} = (1-\alpha)\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{l}\right) + \alpha H^{0}

    Where :math:`\tilde{A}=A+I`

    Args:
        k (int): Number of iters.
        alpha (float): Transmission probability.
        edge_drop (float): The drop rate on the edge of messages received by each node. Default: 1.0.

    Inputs:
        - **x** (Tensor): The input node features. The shape is :math:`(N,*)` where :math:`N` is the number of nodes,
          and :math:`*` could be of any shape.
        - **in_deg** (Tensor): In degree for nodes. In degree for nodes. The shape is :math:`(N, )` where :math:`N` is
          the number of nodes.
        - **out_deg** (Tensor): Out degree for nodes. Out degree for nodes. The shape is :math:`(N, )`
          where :math:`N` is the number of nodes.
        - **g** (Graph): The input graph.

    Outputs:
        Tensor, the output feature of shape :math:`(N,*)` where :math:`*` should be the same as input shape.

    Raises:
        TypeError: If `k` is not an int.
        TypeError: If `alpha` or `edge_drop` is not a float.
        ValueError: If `alpha` is not in range [0.0, 1.0]
        ValueError: If `edge_drop` is not in range (0.0, 1.0]

    Examples:
        >>> import mindspore as ms
        >>> from mindspore_gl.nn.conv import APPNPConv
        >>> from mindspore_gl import GraphField
        >>> n_nodes = 4
        >>> n_edges = 7
        >>> feat_size = 4
        >>> src_idx = ms.Tensor([0, 1, 1, 2, 2, 3, 3], ms.int32)
        >>> dst_idx = ms.Tensor([0, 0, 2, 1, 3, 0, 1], ms.int32)
        >>> ones = ms.ops.Ones()
        >>> feat = ones((n_nodes, feat_size), ms.float32)
        >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
        >>> in_degree = ms.Tensor([3, 2, 1, 1], ms.int32)
        >>> out_degree = ms.Tensor([1, 2, 1, 2], ms.int32)
        >>> appnpconv = APPNPConv(k=3, alpha=0.5, edge_drop=1.0)
        >>> res = appnpconv(feat, in_degree, out_degree, *graph_field.get_graph())
        >>> print(res.shape)
        (4, 4)
    """

    def __init__(self,
                 k: int,
                 alpha: float,
                 edge_drop=1.0,
                 indptr_backward=None,
                 indices_backward=None) -> None:
        super().__init__()
        self.k_ = Validator.check_positive_int(k, "k", self.cls_name)
        self.alpha_ = Validator.check_is_float(alpha, "alpha", self.cls_name)

        if self.alpha_ < 0.0 or self.alpha_ > 1.0:
            raise ValueError(f"For '{self.cls_name}', the 'alpha' should be a number in range [0.0, 1.0], "
                             f"but got {self.alpha_}.")
        edge_drop = Validator.check_is_float(edge_drop, "edge_drop", self.cls_name)
        if edge_drop <= 0.0 or edge_drop > 1.0:
            raise ValueError(f"For '{self.cls_name}', the 'edge_drop' should be a number in range (0.0, 1.0], "
                             f"but got {edge_drop}.")
        self.edge_drop = ms.nn.Dropout(edge_drop)
        self.min_clip = Tensor(1, ms.int32)
        self.max_clip = Tensor(10000000, ms.int32)
        if indptr_backward is not None and indices_backward is not None:
            self.gather = GatherNet(indptr_backward, indices_backward)
            self.csr_reduce_sum = CSRReduceSumNet(indices_backward)
        else:
            self.gather = ms.ops.gather
            # pylint: disable=protected-access
            self.csr_reduce_sum = ms.ops.operations._csr_ops.CSRReduceSum()

    def construct(self, x, in_deg, out_deg, n_nodes, indptr, indices):
        """
        Construct function for APPNPConv.
        """
        out_deg = ms.ops.clip_by_value(out_deg, self.min_clip, self.max_clip)
        out_deg = ms.ops.Reshape()(ms.ops.Pow()(out_deg, -0.5), ms.ops.Shape()(out_deg) + (1,))
        in_deg = ms.ops.clip_by_value(in_deg, self.min_clip, self.max_clip)
        in_deg = ms.ops.Reshape()(ms.ops.Pow()(in_deg, -0.5), ms.ops.Shape()(in_deg) + (1,))
        feat0 = x
        for _ in range(self.k_):
            u_x = self.gather(x, indices, 0)
            u_in_deg = self.gather(in_deg, indices, 0)
            # graph kernel inserts cudNNUniform for each dropout, which cannot be merged with other nodes,
            # and causes the network to fail on reddit due to insufficient memory.
            # The same goes for the Tensor of Tensor version.
            # edge = self.edge_drop(u_x * u_in_deg)
            edge = u_x * u_in_deg
            v_h = self.csr_reduce_sum(indptr, indices, edge, (n_nodes, n_nodes) + edge.shape[1:], 1)
            v_h = ms.ops.reshape(v_h, (n_nodes,) + v_h.shape[2:])
            v_h = v_h * out_deg
            x = (1 - self.alpha_) * v_h + self.alpha_ * feat0
        return x
