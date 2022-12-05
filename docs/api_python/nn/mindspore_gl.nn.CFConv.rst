mindspore_gl.nn.CFConv
======================

.. py:class:: mindspore_gl.nn.CFConv(node_feat_size: int, edge_feat_size: int, hidden_size: int, out_size: int)

    SchNet中的CFConv。
    来自论文 `SchNet: A continuous-filter convolutional neural network for modeling quantum interactions <https://arxiv.org/abs/1706.08566>`_ 。
    它结合了消息传递中的节点和边特征，并更新节点表示。

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} h_j^{l} \circ W^{(l)}e_ij

    其中 :math:`SPP` 代表：

    .. math::
        \text{SSP}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x)) - \log(\text{shift})

    参数：
        - **node_feat_size** (int) - 节点特征大小。
        - **edge_feat_size** (int) - 边特征大小。
        - **hidden_size** (int) - 隐藏层大小。
        - **out_size** (int) - 输出类大小。

    输入：
        - **x** (Tensor) - 输入节点功能。Shape为 :math:`(N,*)`
          其中 :math:`N` 是节点数， :math:`*` 可以是任何形状。
        - **edge_feats** (Tensor) - 输入边缘特征。Shape为 :math:`(M,*)`
          其中 :math:`M` 是边， :math:`*` 可以是任何形状。
        - **g** (Graph) - 输入图表。

    输出：
        - Tensor，输出节点功能。Shape为 :math:`(N, out_size)` 。

    异常：
        - **TypeError** - 如果 `node_feat_size` 不是正整数。
        - **TypeError** - 如果 `edge_feat_size` 不是正整数。
        - **TypeError** - 如果 `hidden_size` 不是正整数。
        - **TypeError** - 如果 `out_size` 不是正整数。
