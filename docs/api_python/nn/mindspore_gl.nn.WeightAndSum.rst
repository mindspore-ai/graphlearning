mindspore_gl.nn.WeightAndSum
============================

.. py:class:: mindspore_gl.nn.WeightAndSum(in_feat_size)

    计算节点的重要性权重并执行加权和。

    参数：
        - **in_feat_size** (int) - 输入特征大小。

    输入：
        - **x** (Tensor) - 要更新的输入节点特征。shape是 :math:`(N,D)`
          其中 :math:`N` 是节点数，:math:`D` 是节点的特征大小。
        - **g** (BatchedGraph) - 输入图。

    输出：
        - **x** (Tensor) - 图形的输出表示。shape是 :math:`(2,D_{out})`
          其中 :math:`D_{out}` 是节点的特征大小

    异常：
        - **TypeError** - 如果 `in_feat_size` 不是int。
