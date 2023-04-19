mindspore_gl.nn.ASTGCN
======================

.. py:class:: mindspore_gl.nn.ASTGCN(n_blocks: int, in_channels: int, k: int, n_chev_filters: int, n_time_filters: int, time_conv_strides: int, num_for_predict: int, len_input: int, n_vertices: int, normalization: Optional[str] = 'sym', bias: bool = True)

    基于Attention的时空图卷积网络。来自于论文 `Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic
    Flow Forecasting <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_ 。

    参数：
        - **n_blocks** (int) - ASTGCN块数。
        - **in_channels** (int) - 输入节点特征大小。
        - **k** (int) - Chebyshev polynomials的阶。
        - **n_chev_filters** (int) - Chebyshev过滤器的数量。
        - **n_time_filters** (int) - 时间过滤器的数量。
        - **time_conv_strides** (int) - 时间卷积期间的时间步长。
        - **num_for_predict** (int) - 未来要进行的预测数。
        - **len_input** (int) - 输入序列的长度。
        - **n_vertices** (int) - 图中的顶点数。
        - **normalization** (str, 可选) - 图Laplacian的归一化方案。默认值：``'sym'``。

          :math:`(L)` 为归一化的矩阵， :math:`(D)` 为度矩阵， :math:`(A)` 为邻接矩阵， :math:`(I)` 为单元矩阵。

          :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}`

        - **bias** (bool, 可选) - layer是否学习加性偏置。默认值：``True``。

    输入：
        - **x** (Tensor) - 输入节点T个时间段的特征。Shape为 :math:`(B, N, F_{in}, T_{in})`
          其中 :math:`N` 是节点数。
        - **g** (Graph) - 输入图。

    输出：
        - Tensor，输出节点特征，shape为 :math:`(B,N,T_{out})` 。

    异常：
        - **TypeError** - 如果 `n_blocks` 、 `in_channels` 、 `k` 、 `n_chev_filters` 、 `n_time_filters` 、 `time_conv_strides` 、`num_for_predict` 、 `len_input` 或 `n_vertices` 不是正整数。
        - **ValueError** - 如果 `normalization` 不是 ``'sym'`` 。
