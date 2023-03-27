mindspore_gl.nn.ChebConv
========================

.. py:class:: mindspore_gl.nn.ChebConv(in_channels: int, out_channels: int, k: int = 3, bias: bool = True)

    切比雪夫谱图卷积层。来自论文 `Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering <https://arxiv.org/abs/1606.09375>`_ 。

    .. math::
        \mathbf{X}^{\prime} = {\sigma}(\sum_{k=1}^{K} \mathbf{\beta}^{k} \cdot
        \mathbf{T}^{k} (\mathbf{\hat{L}}) \cdot X)

        \mathbf{\hat{L}} = 2 \mathbf{L} / {\lambda}_{max} - \mathbf{I}

    :math:`\mathbf{T}^{k}` 递归计算方式为

    .. math::
        \mathbf{T}^{k}(\mathbf{\hat{L}}) = 2 \mathbf{\hat{L}}\mathbf{T}^{k-1}
        - \mathbf{T}^{k-2}

    其中 :math:`\mathbf{k}` 是1或2

    .. math::
        \mathbf{T}^{0} (\mathbf{\hat{L}}) = \mathbf{I}

        \mathbf{T}^{1} (\mathbf{\hat{L}}) = \mathbf{\hat{L}}

    参数：
        - **in_channels** (int) - 输入节点特征大小。
        - **out_channels** (int) - 输出节点特征大小。
        - **k** (int, 可选) - Chebyshev过滤器大小。默认值：3。
        - **bias** (bool, 可选) - 是否使用偏置。默认值：True。

    输入：
        - **x** (Tensor) - 输入节点功能。Shape为 :math:`(N,D_{in})`
          其中 :math:`N`是节点数， :math:`D_{in}` 应等于参数中的 `in_channels` 。
        - **edge_weight** (Tensor) - 边权重。Shape为 :math:`(N\_e,)`
          其中 :math:`N\_e` 是边的数量。
        - **g** (Graph) - 输入图。

    输出：
        - Tensor，输出节点特征的Shape为 :math:`(N,D_{out})`
          其中 :math:`(D_{out})` 应与参数中的 `out_size` 相等。

    异常：
        - **TypeError** - 如果 `in_channels` 或 `out_channels` 或 `k` 不是int。
        - **TypeError** - 如果 `bias` 不是bool。
