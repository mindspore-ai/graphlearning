mindspore_gl.nn.EGConv
======================

.. py:class:: mindspore_gl.nn.EGConv(in_feat_size: int, out_feat_size: int, aggregators: List[str], num_heads: int = 8, num_bases: int = 4, bias: bool = True)

    高效图卷积。来自论文 `Adaptive Filters and Aggregator Fusion for Efficient Graph Convolutions <https://arxiv.org/abs/2104.01481>`_ 。

    .. math::
        h_i^{(l+1)} = {\LARGE ||}_{h=1}^{H} \sum_{\oplus \in \mathcal{A}} \sum_{b=1}^{B} w_{h,\oplus,b}^{(l)}
        \bigoplus_{j \in \mathcal{N(i)}} W_{b}^{(l)} h_{j}^{(l)}

    :math:`\mathcal{N}(i)` 表示 :math:`i` 的邻居节点，
    :math:`W_{b}^{(l)}` 表示基础权重，
    :math:`\oplus` 表示聚合器，
    :math:`w_{h,\oplus,b}^{(l)}` 表示头部、聚合器和底部的每顶点加权系数。

    参数：
        - **in_feat_size** (int) - 输入节点特征大小。
        - **out_feat_size** (int) - 输出节点特征大小。
        - **aggregators** (List[str]) - 要使用的聚合器。支持的聚合器为 'sum'、'mean'、'max'、'min'、'std'、'var'、'symnorm'。默认值：symnorm。
        - **num_heads** (int, 可选) - 头数 :math:`H` 。必须具有 :math:`out\_feat\_size % num\_heads == 0` 。默认值：8。
        - **num_bases** (int, 可选) - 基础权重数 :math:`B` 。默认值：4。
        - **bias** (bool, 可选) - 是否加入可学习偏置。默认值：True。

    输入：
        - **x** (Tensor) - 输入节点功能。Shape为 :math:`(N,D_{in})`
          其中 :math:`N` 是节点数，:math:`D_{in}` 应等于参数中的 `in_feat_size` 。
        - **g** (Graph) - 输入图表。

    输出：
        - Tensor，输出节点特征的Shape为 :math:`(N,D_{out})`
          其中 :math:`(D_{out})` 应与参数中的 `out_feat_size` 相等。

    异常：
        - **TypeError** - 如果 `in_feat_size` 或 `out_feat_size` 或 `num_heads` 不是正整数。
        - **ValueError** - 如果 `out_feat_size` 不能被 `num_heads` 整除。
        - **ValueError** - 如果 `aggregators`- 不在['sum', 'mean', 'max', 'min', 'symnorm', 'var', 'std']中。
