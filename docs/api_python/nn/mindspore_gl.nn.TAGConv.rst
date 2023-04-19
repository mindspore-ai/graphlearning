mindspore_gl.nn.TAGConv
=======================

.. py:class:: mindspore_gl.nn.TAGConv(in_feat_size: int, out_feat_size: int, num_hops: int = 2, bias: bool = True, activation=None)

    拓扑自适应图卷积层。
    来自论文 `Topology Adaptive Graph Convolutional Networks <https://arxiv.org/pdf/1710.10370.pdf>`_。

    .. math::
        H^{K} = {\sum}_{k=0}^K (D^{-1/2} A D^{-1/2})^{k} X {\Theta}_{k}

    其中 :math:`{\Theta}_{k}` 表示线性权重加不同跳数的结果。

    参数：
        - **in_feat_size** (int) - 输入节点特征大小。
        - **out_feat_size** (int) - 输出节点特征大小。
        - **num_hops** (int, 可选) - 跳数。默认值：2。
        - **bias** (bool, 可选) - 是否使用偏置。默认值：True。
        - **activation** (Cell, 可选) - 激活函数。默认值：None。

    输入：
        - **x** (Tensor) - 输入节点功能。shape是 :math:`(N, D_{in})` ，其中 :math:`N` 是节点数，
          和 :math:`D_{in}` 应等于参数中的 `in_feat_size` 。
        - **in_deg** (Tensor) - 节点的入读。shape为 :math:`(N, )` ，其中 :math:`N` 是节点数。
        - **out_deg** (Tensor) - 节点的出度。shape是 :math:`(N, )` ，其中 :math:`N` 是节点数。
        - **g** (Graph) - 输入图。

    输出：
        - Tensor，shape为 :math:`(N, D_{out})` 的输出节点特征，其中 :math:`(D_{out})` 应与参数中的 `out_feat_size` 相等。

    异常：
        - **TypeError** - 如果 `in_feat_size` 或 `out_feat_size` 或 `num_hops` 不是int。
        - **TypeError** - 如果 `bias` 不是bool。
        - **TypeError** - 如果 `activation` 不是 `mindspore.nn.Cell`。
