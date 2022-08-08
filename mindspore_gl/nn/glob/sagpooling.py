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
from mindspore._checkparam import Validator
from mindspore_gl import BatchedGraph
from mindspore_gl.nn.conv import GCNConv2
from .. import GNNCell


class SAGPooling(GNNCell):
    r"""The self-attention pooling operator from the `"Self-Attention Graph
    Pooling" <https://arxiv.org/abs/1904.08082>`_ and `"Understanding
    Attention and Generalization in Graph Neural Networks"
    <https://arxiv.org/abs/1905.02850>`_ papers

    if :obj:`min_score` :math:`\tilde{\alpha}` is :obj:`None`:

        .. math::
            \mathbf{y} &= \textrm{GNN}(\mathbf{X}, \mathbf{A})

            \mathbf{i} &= \mathrm{top}_k(\mathbf{y})

            \mathbf{X}^{\prime} &= (\mathbf{X} \odot
            \mathrm{tanh}(\mathbf{y}))_{\mathbf{i}}

            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}}

    Args:
        in_channels (int): Size of each input sample.
        GNN (GNNCell): A graph neural network layer for calculating projection scores. only GCNConv2 is supported.
            (default: :obj:`mindspore_gl.nn.conv.GCNConv2`)
        activation (Cell): The nonlinearity to use.
            (default: :obj:`mindspore.nn.Tanh`)
        multiplier (Float): A scalar for scaling node feature.
            (default: :obj:1.0)

    Inputs:
        - **x** (Tensor) - The input node features to be updated. The shape is :math:`(N, D)`
          where :math:`N` is the number of nodes,
          and :math:`D` is the feature size of nodes, when attn=None, `D` should be equal to `in_feat_size` in `Args`.
        - **attn** (Tensor) - The input node features for calculating projection score. The shape is :math:`(N, D_{in})`
          where :math:`N` is the number of nodes,
          and :math:`D_{in}` should be equal to `in_feat_size` in `Args`.
          attn can be None, if x is expected to be used for calculating projection score.
        - **node_num** (Int) - total number of nodes in g.
        - **perm_num** (Int) - expected k for topk nodes filtering.
        - **g** (BatchedGraph) - The input graph.

    Outputs:
        - **x** (Tensor) - The updated node features. The shape is :math: `M, D_{out}`
          where :math:`M` equals to `perm_num` in `Inputs`
          and :math: `D_{out}` equals to `D` in `Inputs`.
        - **src_perm** (Tensor) - The updated src nodes.
        - **dst_perm** (Tensor) - The updated dst nodes.
        - **perm** (Tensor) - The node index for topk nodes before updating node index. The shape is :math: `M`
          where :math:`M` equals to `perm_num` in `Inputs`.
        - **perm_score (Tensor) - The projection score for updated nodes.

    Raises:
        TypeError: If `in_feat_size` or `out_size` is not an int.

    Supported Platforms:
         ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore_gl.nn.conv import GCNConv2
        >>> from mindspore_gl.nn.glob import SAGPooling
        >>> from mindspore_gl import BatchedGraphField
        >>> n_nodes = 4
        >>> perm_nodes = 2
        >>> n_edges = 7
        >>> feat_size = 4
        >>> src_idx = ms.Tensor([0, 1, 1, 2, 2, 3, 3], ms.int32)
        >>> dst_idx = ms.Tensor([0, 0, 2, 1, 3, 0, 1], ms.int32)
        >>> ones = ms.ops.Ones()
        >>> feat = ones((n_nodes, feat_size), ms.float32)
        >>> ver_subgraph_idx = ms.Tensor([0, 0, 0, 0], ms.int32)
        >>> edge_subgraph_idx = ms.Tensor([0, 0, 0, 0, 0, 0, 0], ms.int32)
        >>> graph_mask = ms.Tensor([1], ms.int32)
        >>> graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges,
        >>>                                 ver_subgraph_idx, edge_subgraph_idx, graph_mask)
        >>> sagpooling = SAGPooling(in_feat_size=4, GNN = GCNConv2)
        >>> res = gcnconv(feat, None, n_nodes, perm_nodes,  *batched_graph_field.get_batched_graph())
        >>> print(res.shape)
        (2, 4)
    """

    def __init__(self,
                 in_channels: int,
                 GNN=GCNConv2,
                 activation=ms.nn.Tanh,
                 multiplier=1.0) -> None:
        super().__init__()
        self.in_channels = Validator.check_positive_int(in_channels, "in_channels", self.cls_name)
        if GNN is not GCNConv2:
            raise NotImplementedError(f"For '{self.cls_name}', only GCNConv2 as GNN is supported, "
                                      f"but got {GNN}.")
        self.gnn = GNN(in_channels, 1)
        self.multiplier = Validator.check_is_float(multiplier, "multiplier", self.cls_name)
        self.activation = ms.nn.Tanh if activation is None else activation
        self.expand_dims = ms.ops.ExpandDims()
        self.masked_select = ms.ops.MaskedSelect()

    # pylint: disable=arguments-differ
    def construct(self, x, attn, node_num, perm_num, g: BatchedGraph):
        """
        Construct function for SAGPooling.
        """
        attn = x if attn is None else attn
        attn = self.expand_dims(attn, -1) if attn.ndim == 1 else attn
        score = self.gnn(attn, g)
        perm_score, perm = g.topk_nodes(score.astype(ms.float32), perm_num, 0)
        perm_score = self.activation()(perm_score)
        perm_score = perm_score.view((perm_score.size, 1))
        x = perm_score * x[perm]
        x = self.multiplier * x
        mask = ms.numpy.full(node_num, -1.).astype(ms.float32)
        new_node_index = ms.numpy.arange(perm_num, dtype=ms.float32)
        mask[perm] = new_node_index
        row, col = g.src_idx, g.dst_idx
        new_row, new_col = mask[row], mask[col]
        row_mask = (new_row >= 0)
        col_mask = (new_col >= 0)
        mask = ms.ops.logical_and(row_mask, col_mask)
        src_perm = self.masked_select(new_row.view(-1), mask)
        dst_perm = self.masked_select(new_col.view(-1), mask)
        src_perm = src_perm.astype(ms.int32)
        dst_perm = dst_perm.astype(ms.int32)
        return x, src_perm, dst_perm, perm, perm_score
