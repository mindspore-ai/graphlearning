mindspore_gl.nn.SAGPooling
==========================

.. py:class:: mindspore_gl.nn.SAGPooling(in_channels: int, GNN=GCNConv2, activation=ms.nn.Tanh, multiplier=1.0)

    基于self-attention的池化操作。来自 `Self-Attention Graph Pooling <https://arxiv.org/abs/1904.08082>`_ 和
    `Understanding Attention and Generalization in Graph Neural Networks <https://arxiv.org/abs/1905.02850>`_ 。

    .. math::
        \mathbf{y} &= \textrm{GNN}(\mathbf{X}, \mathbf{A})

        \mathbf{i} &= \mathrm{top}_k(\mathbf{y})

        \mathbf{X}^{\prime} &= (\mathbf{X} \odot
        \mathrm{tanh}(\mathbf{y}))_{\mathbf{i}}

        \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}}

    参数：
        - **in_channels** (int) - 每个输入样本的大小。
        - **GNN** (GNNCell, 可选) - 用于计算投影分数的图神经网络层，仅支持GCNConv2。默认值：mindspore_gl.nn.con.GCNConv2。
        - **activation** (Cell, 可选) - 非线性激活函数。默认值：mindspore.nn.Tanh。
        - **multiplier** (float, 可选) - 用于缩放节点功能的标量。默认值：1.0。

    输入：
        - **x** (Tensor) - 要更新的输入节点特征。Shape为 :math:`(N, D)`
          其中 :math:`N` 是节点数， :math:`D` 是节点的特征大小，当 `attn` 为None时，`D` 应等于参数中的 `in_feat_size` 。
        - **attn** (Tensor) - 用于计算投影分数的输入节点特征。Shape为 :math:`(N, D_{in})`
          其中 :math:`N` 是节点数， :math:`D_{in}` 应等于参数中的 `in_feat_size` 。
          如果用 `x` 计算投影分数， `attn` 可以为None。
        - **node_num** (Int) - 以图g中的节点总数。
        - **perm_num** (Int) - Topk个节点过滤中k值。
        - **g** (BatchedGraph) - 输入图。

    输出：
        - **x** (Tensor) - 更新的节点特征。Shape为 :math:`(2, M, D_{out})`
          其中 :math:`M` 等于 `Inputs` 中的 `perm_num` 和 :math:`D_{out}` 等于 `Inputs` 中的 `D` 。
        - **src_perm** (Tensor) - 更新的src节点。
        - **dst_perm** (Tensor) - 更新的dst节点。
        - **perm** (Tensor) - 更新节点索引之前topk节点的节点索引。Shape为 :math:`M`，其中 :math:`M` 等于 `Inputs` 中的 `perm_num` 。
        - **perm_score** (Tensor) - 更新节点的投影分数。

    异常：
        - **TypeError** - 如果 `in_feat_size` 或 `out_size` 不是int。
