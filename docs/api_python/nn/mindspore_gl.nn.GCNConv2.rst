mindspore_gl.nn.GCNConv2
========================

.. py:class:: mindspore_gl.nn.GCNConv2(in_feat_size: int, out_size: int) -> None

    图卷积网络层。
    来自论文 `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`_ 。

    .. math::
        h_i^{(l+1)} = (\sum_{j\in\mathcal{N}(i)}h_j^{(l)}W_1^{(l)}+b^{(l)} )+h_i^{(l)}W_2^{(l)}

    :math:`\mathcal{N}(i)` 表示 :math:`i` 的邻居节点。
    :math:`W_1` 和 `W_2` 对应邻居节点和根节点的fc层。

    参数：
        - **in_feat_size** (int) - 输入节点特征大小。
        - **out_size** (int) - 输出节点特征大小。

    输入：
        - **x** (Tensor) - 输入节点特征。Shape为 :math:`(N,D_{in})`
          其中 :math:`N` 是节点数， :math:`D_{in}` 应等于 `Args` 中的 `in_feat_size` 。
        - **g** (Graph) - 输入图。

    输出：
        - Tensor，Shape为 :math:`(N,D_{out})` 的输出节点特征，其中 :math:`(D_{out})` 应与 `Args` 中的 `out_size` 。

    异常：
        - **TypeError** - 如果 `in_feat_size` 或 `out_size` 不是int。
