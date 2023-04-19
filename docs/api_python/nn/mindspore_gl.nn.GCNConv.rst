mindspore_gl.nn.GCNConv
=======================

.. py:class:: mindspore_gl.nn.GCNConv(in_feat_size: int, out_size: int, activation=None, dropout=0.5)

    图卷积网络层。来自论文 `Semi-Supervised Classification with Graph Convolutional Networks <https://arxiv.org/abs/1609.02907>`_ 。

    .. math::
        h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ji}}h_j^{(l)}W^{(l)})

    :math:`\mathcal{N}(i)` 表示 :math:`i` 的邻居节点。
    :math:`c_{ji} = \sqrt{|\mathcal{N}(j)|}\sqrt{|\mathcal{N}(i)|}` 。

    .. math::
        h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{e_{ji}}{c_{ji}}h_j^{(l)}W^{(l)})

    参数：
        - **in_feat_size** (int) - 输入节点特征大小。
        - **out_size** (int) - 输出节点特征大小。
        - **activation** (Cell, 可选) - 激活函数。默认值：``None``。
        - **dropout** (float, 可选) - dropout rate，大于等于0，小于1。例如，dropout=0.1，抛弃10%的输入单元。默认值：``0.5``。

    输入：
        - **x** (Tensor) - 输入节点功能。Shape为 :math:`(N, D_{in})`
          其中 :math:`N` 是节点数， :math:`D_{in}` 应等于参数中的 `in_feat_size` 。
        - **in_deg** (Tensor) - 节点的入度。Shape为 :math:`(N, )` 其中 :math:`N` 是节点数。
        - **out_deg** (Tensor) - 节点的出度。Shape为 :math:`(N,)` 。
          其中 :math:`N` 是节点数。
        - **g** (Graph) - 输入图。

    输出：
        - Tensor，输出节点特征的Shape为 :math:`(N, D_{out})` ，其中 :math:`(D_{out})` 应与参数中的 `out_size` 相等。

    异常：
        - **TypeError** - 如果 `in_feat_size` 或 `out_size` 不是int。
        - **TypeError** - 如果 `dropout` 不是float。
        - **TypeError** - 如果 `activation` 不是 `mindspore.nn.Cell`。
        - **ValueError** - 如果 `dropout` 不在(0.0, 1.0]范围。
