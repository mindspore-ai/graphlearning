mindspore_gl.nn.EDGEConv
========================

.. py:class:: mindspore_gl.nn.EDGEConv(in_feat_size: int, out_feat_size: int, batch_norm: bool, bias=True)

    EdgeConv层。来自论文 `Dynamic Graph CNN for Learning on Point Clouds <https://arxiv.org/pdf/1801.07829>`_ 。

    .. math::
        h_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} (
       \Theta \cdot (h_j^{(l)} - h_i^{(l)}) + \Phi \cdot h_i^{(l)})

    :math:`\mathcal{N}(i)` 表示 :math:`i` 的邻居节点。
    :math:`\Theta` 和 :math:`\Phi` 表示线性层。

    参数：
        - **in_feat_size** (int) - 输入节点特征大小。
        - **out_feat_size** (int) - 输出节点特征大小。
        - **batch_norm** (bool) - 是否使用批处理归一化。
        - **bias** (bool) - 是否使用偏差。默认值：True。

    输入：
        - **x** (Tensor) - 输入节点特征。Shape为 :math:`(N,*)`
          其中 :math:`N` 是节点数， :math:`*` 可以是任何形状。
        - **g** (Graph) - 输入图。

    输出：
        Tensor，输出节点特征。Shape为 :math:`(N, out_feat_size)` 。

    异常：
        TypeError：如果 `in_feat_size` 不是正整数。
        TypeError：如果 `out_feat_size` 不是正整数。
        TypeError：如果 `batch_norm` 不是bool。
        TypeError：如果 `bias` 不是bool。
