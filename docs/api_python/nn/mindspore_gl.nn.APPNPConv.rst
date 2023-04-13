mindspore_gl.nn.APPNPConv
=========================

.. py:class:: mindspore_gl.nn.APPNPConv(k: int, alpha: float, edge_drop=0.0)

    神经预测层中的近似个性化传播。
    来自论文 `Predict then Propagate: Graph Neural Networks meet Personalized PageRank <https://arxiv.org/pdf/1810.05997.pdf>`_ 。

    .. math::
        H^{0} = X \\
        H^{l+1} = (1-\alpha)\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{l}\right) + \alpha H^{0}

    其中 :math:`\tilde{A}=A+I`

    参数：
        - **k** (int) - 迭代次数。
        - **alpha** (float) - 传输概率。
        - **edge_drop** (float, 可选) - 每个节点接收到的边消息的dropout rate。默认值：0.0。

    输入：
        - **x** (Tensor) - 输入节点功能。Shape为 :math:`(N,*)`
          其中 :math:`N` 是节点数， :math:`*` 可以是任何shape。
        - **in_deg** (Tensor) - 节点的入度。Shape为 :math:`(N,)`
          其中 :math:`N` 是节点数。
        - **out_deg** (Tensor) - 节点的出度。Shape为 :math:`(N, )`
          其中 :math:`N` 是节点数。
        - **g** (Graph) - 输入图表。

    输出：
        - Tensor，输出特征Shape为 :math:`(N,*)` ，其中 :math:`*` 应与输入shape相同。

    异常：
        - **TypeError** - 如果 `k` 不是int。
        - **TypeError** - 如果 `alpha` 或 `edge_drop` 不是float。
        - **ValueError** - 如果 `alpha` 不在范围[0.0, 1.0]内。
        - **ValueError** - 如果 `edge_drop` 不在范围[0.0, 1.0)内。
