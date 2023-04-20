mindspore_gl.nn.GMMConv
=======================

.. py:class:: mindspore_gl.nn.GMMConv(in_feat_size: int, out_feat_size: int, coord_dim: int, n_kernels: int, residual=False, bias=False, aggregator_type='sum')

    高斯混合模型卷积层。
    来自论文 `Geometric Deep Learning on Graphs and Manifolds using Mixture Model CNNs <http://openaccess.thecvf.com/content_cvpr_2017/papers/Monti_Geometric_Deep_Learning_CVPR_2017_paper.pdf>`_ 。

    .. math::
        u_{ij} = f(x_i, x_j), x_j \in \mathcal{N}(i) \\
        w_k(u) = \exp\left(-\frac{1}{2}(u-\mu_k)^T \Sigma_k^{-1} (u - \mu_k)\right) \\
        h_i^{l+1} = \mathrm{aggregate}\left(\left\{\frac{1}{K}
         \sum_{k}^{K} w_k(u_{ij}), \forall j\in \mathcal{N}(i)\right\}\right)

    其中 :math:`u` 表示顶点与它其中一个邻居之间的伪坐标，使用
    函数 :math:`f` ，其中 :math:`\Sigma_k^{-1}` 和 :math:`\mu_k` 是协方差的可学习参数矩阵和高斯核的均值向量。

    参数：
        - **in_feat_size** (int) - 输入节点特征大小。
        - **out_feat_size** (int) - 输出节点特征大小。
        - **coord_dim** (int) - 伪坐标的维度。
        - **n_kernels** (int) - 内核数。
        - **residual** (bool, 可选) - 是否使用残差。默认值：``False``。
        - **bias** (bool, 可选) - 是否使用偏置。默认值：``False``。
        - **aggregator_type** (str, 可选) - 聚合器的类型。默认值：``'sum'``。

    输入：
        - **x** (Tensor) - 输入节点特征。Shape为 :math:`(N, D_{in})` ，其中 :math:`N` 是节点数， :math:`D_{in}` 应等于参数中的 `in_feat_size` 。
        - **pseudo** (Tensor) - 伪坐标张量。
        - **g** (Graph) - 输入图。

    输出：
        - Tensor，Shape为 :math:`(N, D_{out})`的输出节点特征，其中 :math:`(D_{out})` 应等于参数中的 `out_size`。

    异常：
        - **SyntaxError** - 当 `aggregation_type` 不等于 ``'sum'`` 时。
        - **TypeError** - 如果 `in_feat_size` 或 `out_feat_size` 或 `coord_dim` 或 `n_kernels` 不是int。
        - **TypeError** - 如果 `bias` 或 `resual` 不是bool。
