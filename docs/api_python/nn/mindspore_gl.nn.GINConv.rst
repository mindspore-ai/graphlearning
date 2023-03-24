mindspore_gl.nn.GINConv
=======================

.. py:class:: mindspore_gl.nn.GINConv(activation, init_eps=0., learn_eps=False, aggregation_type="sum")

    图同构网络层。
    从论文 `How Powerful are Graph Neural Networks? <https://arxiv.org/pdf/1810.00826.pdf>`_ 。

    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \mathrm{aggregate}\left(\left\{h_j^{l}, j\in\mathcal{N}(i)
        \right\}\right)\right)

    如果提供了各个边权重，则加权图卷积定义为：

    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \mathrm{aggregate}\left(\left\{e_{ji} h_j^{l}, j\in\mathcal{N}(i)
        \right\}\right)\right)

    参数：
        - **activation** (mindspore.nn.Cell) - 激活函数。
        - **init_eps** (float, 可选) - eps的初始化值。默认值：0.0。
        - **learn_eps** (bool, 可选) - eps是否可学习。默认值：False。
        - **aggregation_type** (str, 可选) - 聚合类型，应在'sum'、'max'和'avg'中。默认值：sum。

    输入：
        - **x** (Tensor) - 输入节点特征。Shape为 :math:`(N,*)` 其中 :math:`N` 是节点数， :math:`*` 可以是任何shape。
        - **edge_weight** (Tensor) - 输入边权重。Shape为 :math:`(M,*)` ，其中 :math:`M` 是数字节点， :math:`*` 可以是任何shape。
        - **g** (Graph) - 输入图。

    输出：
        - Tensor，输出节点特征。Shape为 :math:`(N,out\_feat\_size)` 。

    异常：
        - **TypeError** - 如果 `activation` 不是mindspore.nn.Cell。
        - **TypeError** - 如果 `init_eps` 不是float。
        - **TypeError** - 如果 `learn_eps` 不是bool值。
        - **SyntaxError** - 当 `aggregation_type` 不在'sum'、'max'和'avg'中时引发。
