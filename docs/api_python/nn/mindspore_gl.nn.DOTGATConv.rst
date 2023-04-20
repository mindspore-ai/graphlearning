mindspore_gl.nn.DOTGATConv
==========================

.. py:class:: mindspore_gl.nn.DOTGATConv(in_feat_size: int, out_feat_size: int, num_heads: int, bias=False)

    在GAT中应用点积版的self-attention。
    来自论文 `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`_ 。

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i, j} h_j^{(l)}

    :math:`\alpha_{i, j}` 表示节点 :math:`i` 和节点 :math:`j` 之间的attention分数。

    .. math::
        \alpha_{i, j} = \mathrm{softmax_i}(e_{ij}^{l}) \\
        e_{ij}^{l} = ({W_i^{(l)} h_i^{(l)}})^T \cdot {W_j^{(l)} h_j^{(l)}}

    参数：
        - **in_feat_size** (int) - 输入节点特征大小。
        - **out_feat_size** (int) - 输出节点特征大小。
        - **num_heads** (int) - GAT中使用的attention头数。
        - **bias** (bool, 可选) - 是否使用偏置。默认值：``False``。

    输入：
        - **x** (Tensor) - 输入节点特征。Shape为 :math:`(N,*)` ，其中 :math:`N` 是节点数， :math:`*` 可以是任何shape。
        - ***g** (Graph) - 输入图。

    输出：
        - Tensor，输出节点特征。Shape为 :math:`(N, num\_heads, out\_feat\_size)` 。

    异常：
        - **TypeError** - 如果 `in_feat_size` 不是正整数。
        - **TypeError** - 如果 `out_feat_size` 不是正整数。
        - **TypeError** - 如果 `num_heads` 不是正整数。
        - **TypeError** - 如果 `bias` 不是bool。
