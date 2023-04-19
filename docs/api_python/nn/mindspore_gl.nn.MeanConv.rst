mindspore_gl.nn.MeanConv
========================

.. py:class:: mindspore_gl.nn.MeanConv(in_feat_size: int, out_feat_size: int, feat_drop=0.4, bias=False, norm=None, activation=None)

    GraphSAGE层。来自论文 `Inductive Representation Learning on Large Graphs <https://arxiv.org/pdf/1706.02216.pdf>`_。

    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} = \mathrm{aggregate}
        \left(\{h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right) \\

        h_{i}^{(l+1)} = \sigma \left(W \cdot \mathrm{concat}
        (h_{i}^{l}, h_{\mathcal{N}(i)}^{l+1}) \right)\\

        h_{i}^{(l+1)} = \mathrm{norm}(h_{i}^{l})

    如果提供了各个边的权重，则加权图卷积定义为：

    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} = \mathrm{aggregate}
        \left(\{e_{ji} h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)

    参数：
        - **in_feat_size** (int) - 输入节点特征大小。
        - **out_feat_size** (int) - 输出节点特征大小。
        - **feat_drop** (float, 可选) - dropout rate，大于等于0，小于1。例如，feat_drop=0.1，抛弃10%的输入单元。默认值：``0.4``。
        - **bias** (bool, 可选) - 是否使用偏置。默认值：``False``。
        - **norm** (Cell, 可选) - 归一化函数单元。默认值：``None``。
        - **activation** (Cell, 可选) - 激活函数Cell。默认值：``None``。

    输入：
        - **x** (Tensor) - 输入节点特征。Shape为 :math:`(N,D\_in)`
          其中 :math:`N` 是节点数， :math:`D\_in` 可以是任何shape。
        - **self_idx** (Tensor) - 节点id。Shape为 :math:`(N\_v,)`
          其中 :math:`N\_v` 是自节点的数量。
        - **g** (Graph) - 输入图。

    输出：
        - Tensor，输出特征shape为 :math:`(N\_v,D\_out)` 。
          其中 :math:`N\_v` 是自节点的数量， :math:`D\_out` 可以是任何shape。

    异常：
        - **TypeError** - 如果 `in_feat_size` 或 `out_feat_size` 不是int。
        - **TypeError** - 如果 `bias` 不是bool。
        - **TypeError** - 如果 `norm` 不是 `mindspore.nn.Cell`。
        - **ValueError** - 如果 `dropout` 不在范围[0.0, 1.0)内。
        - **ValueError** - 如果 `activation` 不是 ``tanh`` 或 ``relu``。
