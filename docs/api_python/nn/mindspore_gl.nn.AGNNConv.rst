mindspore_gl.nn.AGNNConv
========================

.. py:class:: mindspore_gl.nn.AGNNConv(init_beta: float = 1.0, learn_beta: bool = True)

    基于Attention的图神经网络。来自论文 `Attention-based Graph Neural Network for Semi-Supervised Learning <https://arxiv.org/abs/1803.03735>`_ 。

    .. math::
        H^{l+1} = P H^{l}

    计算 :math:`P` :

    .. math::
        P_{ij} = \mathrm{softmax}_i ( \beta \cdot \cos(h_i^l, h_j^l))

    :math:`\beta` 是单个标量参数。

    参数：
        - **init_beta** (float) - 初始化 :math:`\beta` ，单个标量参数。默认值：1.0。
        - **learn_beta** (bool) - 是否 :math:`\beta` 可学习。默认值：True。

    输入：
        - **x** (Tensor) - 输入节点特征。Shape为 :math:`(N,*)` ，其中 :math:`N` 是节点数，
          :math:`*` 可以是任何形状。
        - **g** (Graph) - 输入图表。

    输出：
        - Tensor，输出节点特征，其中shape应与输入 `x` 相同。

    异常：
        - **TypeError** - 如果 `init_beta` 不是float。
        - **TypeError** - 如果 `learn_beta` 不是bool。
