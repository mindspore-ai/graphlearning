mindspore_gl.nn.GATv2Conv
=========================

.. py:class:: mindspore_gl.nn.GATv2Conv(in_feat_size: int, out_size: int, num_attn_head: int, input_drop_out_rate: float = 1.0, attn_drop_out_rate: float = 1.0, leaky_relu_slope: float = 0.2, activation=None, add_norm=False)

    图 Attention 网络v2。来自论文 `How Attentive Are Graph Attention Networks?
    <https://arxiv.org/pdf/2105.14491.pdf>`_ ，它修复了GATv2的静态attention问题。

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}

    :math:`\alpha_{i, j}` 表示节点 :math:`i` 和节点 :math:`j` 之间的attention score。

    .. math::
        \alpha_{ij}^{l} = \mathrm{softmax_i} (e_{ij}^{l}) \\
        e_{ij}^{l} = \vec{a}^T \mathrm{LeakyReLU}\left(W [h_{i} \| h_{j}]\right)

    参数：
        - **in_feat_size** (int) - 输入节点特征大小。
        - **out_size** (int) - 输出节点特征大小。
        - **num_attn_head** (int) - GATv2中使用的attention头数。
        - **input_drop_out_rate** (float, 可选) - 输入丢弃的keep rate。默认值：1.0。
        - **attn_drop_out_rate** (float, 可选) - attention丢弃的keep rate。默认值：1.0。
        - **leaky_relu_slope** (float, 可选) - leaky relu的斜率。默认值：0.2。
        - **activation** (Cell, 可选) - 激活函数，默认值：None。
        - **add_norm** (bool, 可选) - 边信息是否需要归一化。默认值：False。

    输入：
        - **x** (Tensor) - 输入节点功能。Shape为 :math:`(N,D_{in})`
          其中 :math:`N` 是节点数， :math:`D_{in}` 可以是任何shape。
        - **g** (Graph) - 输入图。

    输出：
        - Tensor，输出特征Shape为 :math:`(N,D_{out})` 其中 :math:`D_{out}` 应等于
          :math:`D_{in}*num\_attn\_head` 。

    异常：
        - **TypeError** - 如果 `in_feat_size` 、 `out_size` 或 `num_attn_head` 不是int。
        - **TypeError** - 如果 `input_drop_out_rate` 、 `attn_drop_out_rate` 或 `leaky_relu_slope` 不是float。
        - **TypeError** - 如果 `activation` 不是mindspore.nn.Cell。
        - **ValueError** - 如果 `input_drop_out_rate` 或 `attn_drop_out_rate` 不在范围(0.0, 1.0]内。
