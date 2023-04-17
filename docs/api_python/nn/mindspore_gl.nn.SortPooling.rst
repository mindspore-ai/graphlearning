mindspore_gl.nn.SortPooling
===========================

.. py:class:: mindspore_gl.nn.SortPooling(k)

    将排序池化应用于图形中的节点。
    来自论文 `End-to-End Deep Learning Architecture for Graph Classification <https://muhanzhang.github.io/papers/AAAI_2018_DGCNN.pdf>`_ 。

    排序池化首先将节点特征沿特征维度升序排序。
    然后选择Topk节点的排名特征（按每个节点的最大值排序）。

    参数：
        - **k** (int) - 每个图保留的节点数。

    输入：
        - **x** (Tensor) - 要更新的输入节点特征。Shape为 :math:`(N, D)`，
          其中 :math:`N` 是节点数， :math:`D` 是节点的特征大小。
        - **g** (BatchedGraph) - 输入图。

    输出：
        - **x** (Tensor) - 图形的输出表示。Shape为 :math:`(2, D_{out})`
          其中 :math:`D_{out}` 是节点的特征的双倍大小。

    异常：
        - **TypeError** - 如果 `k` 不是int。
