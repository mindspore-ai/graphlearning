mindspore_gl.nn.STConv
======================

.. py:class:: mindspore_gl.nn.STConv(num_nodes: int, in_channels: int, hidden_channels: int, out_channels: int, kernel_size: int = 3, k: int = 3, bias: bool = True)

    时空图卷积层。
    来自论文 `A deep learning framework for traffic forecasting
    arXiv preprint arXiv:1709.04875, 2017. <https://arxiv.org/pdf/1709.04875.pdf>`_ 。
    STGCN层包含2个时间卷积层和1个图卷积层（ChebyNet）。

    参数：
        - **num_nodes** (int) - 节点数。
        - **in_channels** (int) - 输入节点特征大小。
        - **hidden_channels** (int) - 隐藏特征大小。
        - **out_channels** (int) - 输出节点特征大小。
        - **kernel_size** (int) - 卷积内核大小。默认值：3。
        - **k** (int) - Chebyshev过滤器大小。默认值：3。
        - **bias** (bool) - 是否使用偏差。默认值：True。

    输入：
        - **x** (Tensor) - 输入节点特征。Shape为 :math:`(B, T, N, (D_{in}))`
          其中 :math:`B` 是批处理的大小， :math:`T` 是输入时间步数，
          :math:`N` 是节点数。
          :math:`(D_{in})` 应等于 `Args` 中的 `in_channels` 。
        - **edge_weight** (Tensor) - 边缘权重。Shape为 :math:`(N\_e,)`
          其中 :math:`N\_e` 是边的数量。
        - **g** (Graph) - 输入图。

    输出：
        - Tensor，输出节点特征，shape为 :math:`(B,D_{out},N,T)`，
          其中 :math:`B` 是批处理的大小， :math:`(D_{out})` 应与
          `Args` 中的 `out_channels` ， :math:`N` 是节点数，
          :math:`T` 是输入时间步数。

    异常：
        - **TypeError** - 如果 `num_nodes` 、 `in_channels` 、 `out_channels` 、 `hidden_channels` 、 `kernel_size` 、 `k` 不是int。
        - **TypeError** - 如果 `bias` 不是bool。
