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
"""SAGPooling Layer"""
# pylint: disable=unused-import
import mindspore as ms
from mindspore import dtype as mstype
from mindspore_gl import BatchedGraph
from mindspore_gl.nn.conv import GCNConv2
from .. import GNNCell

class SAGPooling(GNNCell):
    r"""
    The self-attention pooling operator. From the `Self-Attention Graph
    Pooling <https://arxiv.org/abs/1904.08082>`_ and `Understanding
    Attention and Generalization in Graph Neural Networks
    <https://arxiv.org/abs/1905.02850>`_ papers.

    .. math::
        \mathbf{y} &= \textrm{GNN}(\mathbf{X}, \mathbf{A})

        \mathbf{i} &= \mathrm{top}_k(\mathbf{y})

        \mathbf{X}^{\prime} &= (\mathbf{X} \odot
        \mathrm{tanh}(\mathbf{y}))_{\mathbf{i}}

        \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}}

    Args:
        in_channels (int): Size of each input sample.
        GNN (GNNCell, optional): A graph neural network layer for calculating projection scores. only GCNConv2
            is supported. Default: `mindspore_gl.nn.conv.GCNConv2`.
        activation (Cell, optional): The nonlinearity activation function Cell to use. Default: `mindspore.nn.Tanh`.
        multiplier (float, optional): A scalar for scaling node feature. Default: ``1``.

    Inputs:
        - **x** (Tensor) - The input node features to be updated. The shape is :math:`(N, D)`
          where :math:`N` is the number of nodes,
          and :math:`D` is the feature size of nodes, when `attn` is None, `D` should be equal to `in_feat_size` in
          `Args`.
        - **attn** (Tensor) - The input node features for calculating projection score. The shape is :math:`(N, D_{in})`
          where :math:`N` is the number of nodes,
          and :math:`D_{in}` should be equal to `in_feat_size` in `Args`.
          attn can be None, if x is expected to be used for calculating projection score.
        - **node_num** (Int) - total number of nodes in g.
        - **perm_num** (Int) - expected k for topk nodes filtering.
        - **g** (BatchedGraph) - The input graph.

    Outputs:
        - **x** (Tensor) - The updated node features. The shape is :math:`(2, M, D_{out})`,
          where :math:`M` equals to `perm_num` in `Inputs`,
          and :math:`D_{out}` equals to `D` in `Inputs`.
        - **src_perm** (Tensor) - The updated source nodes.
        - **dst_perm** (Tensor) - The updated destination nodes.
        - **perm** (Tensor) - The node index for topk nodes before updating node index. The shape is :math:`M`,
          where :math:`M` equals to `perm_num` in `Inputs`.
        - **perm_score** (Tensor) - The projection score for updated nodes.

    Raises:
        TypeError: If `in_feat_size` or `out_size` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore_gl.nn import SAGPooling
        >>> from mindspore_gl import BatchedGraphField
        >>> node_feat = ms.Tensor([[1, 2, 3, 4], [2, 4, 1, 3], [1, 3, 2, 4],
        ...                        [9, 7, 5, 8], [8, 7, 6, 5], [8, 6, 4, 6], [1, 2, 1, 1]],
        ...                       ms.float32)
        >>> n_nodes = 7
        >>> n_edges = 8
        >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
        >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
        >>> ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
        >>> edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
        >>> graph_mask = ms.Tensor([0, 1], ms.int32)
        >>> batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx,
        ...                                         edge_subgraph_idx, graph_mask)
        >>> net = SAGPooling(4)
        >>> feature, src, dst, ver_subgraph, edge_subgraph, perm, perm_score = net(node_feat, None, 2,
        ...                                                                    *batched_graph_field.get_batched_graph())
        >>> print(feature.shape)
        (2, 2, 4)
    """

    def __init__(self,
                 in_channels: int,
                 GNN=GCNConv2,
                 activation=ms.nn.Tanh,
                 multiplier=1.0):
        super().__init__()
        assert isinstance(in_channels, int) and in_channels > 0, "in_channels must be positive int"
        assert isinstance(multiplier, float), "multiplier must be float"

        if GNN is not GCNConv2:
            raise NotImplementedError(f"For '{self.cls_name}', only GCNConv2 as GNN is supported, "
                                      f"but got {GNN}.")
        self.gnn = GNN(in_channels, 1)
        self.multiplier = multiplier
        self.activation = ms.nn.Tanh if activation is None else activation
        self.expand_dims = ms.ops.ExpandDims()
        self.masked_select = ms.ops.MaskedSelect()

    # pylint: disable=arguments-differ
    def construct(self, x, attn, perm_num, g: BatchedGraph):
        """
        Construct function for SAGPooling.
        """
        if x.dtype != mstype.float32:
            raise TypeError('Only float32 node features are supported but got ' + str(x.dtype) + ' for input_1')
        if (attn is not None) and (attn.dtype != mstype.float32):
            raise TypeError('Only float32 node features are supported but got ' + str(attn.dtype) + ' for input_2')
        attn = x if attn is None else attn
        attn = self.expand_dims(attn, -1) if attn.ndim == 1 else attn
        score = self.gnn(attn, g)
        perm_score, perm = g.topk_nodes(score.astype(ms.float32), perm_num, 0)
        perm_score = self.activation()(perm_score)
        x = perm_score * x[perm]
        x = self.multiplier * x
        node_num = g.n_nodes
        mask = ms.numpy.full(node_num, -1.).astype(ms.float32)
        perm = perm.view(perm.size)
        new_node_index = ms.numpy.arange(perm.size, dtype=ms.float32)
        ver_subgraph_idx = g.ver_subgraph_idx[perm]
        mask[perm] = new_node_index
        row, col = g.src_idx, g.dst_idx
        new_row, new_col = mask[row], mask[col]
        row_mask = (new_row >= 0)
        col_mask = (new_col >= 0)
        mask = ms.ops.logical_and(row_mask, col_mask)
        src_perm = self.masked_select(new_row.view(-1), mask)
        dst_perm = self.masked_select(new_col.view(-1), mask)
        edge_subgraph_idx = self.masked_select(g.edge_subgraph_idx, mask)
        src_perm = src_perm.astype(ms.int32)
        dst_perm = dst_perm.astype(ms.int32)
        return x, src_perm, dst_perm, ver_subgraph_idx, edge_subgraph_idx, perm, perm_score
