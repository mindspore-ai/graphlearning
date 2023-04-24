mindspore_gl.nn.SGConv
======================

.. py:class:: mindspore_gl.nn.SGConv(in_feat_size: int, out_feat_size: int, num_hops: int = 1, cached: bool = True, bias: bool = True, norm=None)

    简化的图卷积层。
    来自论文 `Simplifying Graph Convolutional Networks <https://arxiv.org/pdf/1902.07153.pdf>`_ 。

    .. math::
        H^{K} = (\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2})^K X \Theta

    其中 :math:`\tilde{A}=A+I` 。

    ..note:
        目前只支持PYNATIVE模式。

    参数：
        - **in_feat_size** (int) - 输入节点特征大小。
        - **out_feat_size** (int) - 输出节点特征大小。
        - **num_hops** (int, 可选) - hop的数量。默认值：``1``。
        - **cached** (bool, 可选) - 是否使用缓存。默认值：``True``。
        - **bias** (bool, 可选) - 是否使用偏置。默认值：``True``。
        - **norm** (Cell, 可选) - 归一化函数Cell。默认值：``None``。

    输入：
        - **x** (Tensor) - 输入节点功能。Shape为 :math:`(N, D_{in})` ，其中 :math:`N` 是节点数， :math:`D_{in}` 应等于参数中的 `in_feat_size` 。
        - **in_deg** (Tensor) - 节点的入度。Shape为 :math:`(N, )` ，其中 :math:`N` 是节点数。
        - **out_deg** (Tensor) - 节点的出度。Shape为 :math:`(N, )`
          其中 :math:`N` 是节点数。
        - **g** (Graph) - 输入图。

    输出：
        - Tensor，Shape为 :math:`(N, D_{out})` 的输出节点特征，其中 :math:`(D_{out})` 应与参数中的 `out_feat_size` 相等。

    异常：
        - **TypeError** - 如果 `in_feat_size` 或 `out_feat_size` 或 `num_hops` 不是int。
        - **TypeError** - 如果 `bias` 或 `cached` 不是bool。
        - **TypeError** - 如果 `norm` 不是 `mindspore.nn.Cell`。
