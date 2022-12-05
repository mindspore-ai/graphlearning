mindspore_gl.nn.SAGEConv
========================

.. py:class:: mindspore_gl.nn.SAGEConv(in_feat_size: int, out_feat_size: int, aggregator_type: str = 'pool', bias=True, norm=None, activation: mindspore.nn.cell.Cell = None)

    GraphSAGE层，来自论文 `Inductive Representation Learning on Large Graphs <https://arxiv.org/pdf/1706.02216.pdf>`_。

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
        - **aggregator_type** (str) - 聚合器的类型，应在'pool'、'lstm'和'mean'中。默认值：pool。
        - **bias** (bool) - 是否使用偏差。默认值：True。
        - **norm** (Cell) - 归一化函数单元。默认值：None。
        - **activation** (Cell) - 激活函数Cell。默认值：None。

    输入：
        - **x** (Tensor) - 输入节点特征。Shape为 :math:`(N,D\_in)`
          其中 :math:`N` 是节点数， :math:`D\_in` 可以是任何形状。
        - **edge_weight** (Tensor) - 边权重。Shape为 :math:`(N\_e,)`
          其中 :math:`N\_e` 是边的数量。
        - **g** (Graph) - 输入图。

    输出：
        - Tensor，输出特征Shape为 :math:`(N,D\_out)`
          其中 :math:`N` 是节点数， :math:`D\_out` 可以是任何形状。

    异常：
        - **KeyError** - 如果 `aggregator` 类型不是pool、lstm或mean。
        - **TypeError** - 如果 `in_feat_size` 或 `out_feat_size` 不是int。
        - **TypeError** - 如果 `bias` 不是bool。
        - **TypeError** - 如果 `activation` 类型不是mindspore.nn.Cell。
        - **TypeError** - 如果 `norm` 类型不是mindspore.nn.Cell。
