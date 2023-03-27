mindspore_gl.nn.NNConv
======================

.. py:class:: mindspore_gl.nn.NNConv(in_feat_size: int, out_feat_size: int, edge_embed, aggregator_type: str = "sum", residual=False, bias=True)

    图卷积层。
    来自论文 `Neural Message Passing for Quantum Chemistry <https://arxiv.org/pdf/1704.01212.pdf>`_ 。

    .. math::
        h_{i}^{l+1} = h_{i}^{l} + \mathrm{aggregate}\left(\left\{
        f_\Theta (e_{ij}) \cdot h_j^{l}, j\in \mathcal{N}(i) \right\}\right)

    其中 :math:`f_\Theta` 是一个具有可学习参数的函数。

    参数：
        - **in_feat_size** (int) - 输入节点特征大小。
        - **out_feat_size** (int) - 输出节点特征大小。
        - **edge_embed** (mindspore.nn.Cell) - 边嵌入函数单元。
        - **aggregator_type** (str, 可选) - 聚合器的类型。默认值：'sum'。
        - **residual** (bool, 可选) - 是否使用残差。默认值：False。
        - **bias** (bool, 可选) - 是否使用偏置。默认值：True。

    输入：
        - **x** (Tensor) - 输入节点特征。shape是 :math:`(N,D\_in)`
          其中 :math:`N` 是节点数， :math:`D\_in` 可以是任何shape。
        - **edge_feat** (Tensor) - 边特征。shape是 :math:`(N\_e,F\_e)`
          其中 :math:`N\_e` 是边的数量， :math:`F\_e` 是边特征的数量。
        - **g** (Graph) - 输入图。

    输出：
        - Tensor，输出特征Shape为 :math:`(N,D\_out)`
          其中 :math:`N` 是节点数， :math:`D\_out` 可以是任何shape。

    异常：
        - **TypeError** - 如果 `edge_embed` 类型不是mindspore.nn.Cell或 `aggregator_type` 不是'sum'。
        - **TypeError** - 如果 `in_feat_size` 或 `out_feat_size` 不是int。
        - **TypeError** - 如果 `residual` 或 `bias` 不是bool。
