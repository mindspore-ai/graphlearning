mindspore_gl.GraphField
=======================

.. py:class:: mindspore_gl.GraphField(src_idx, dst_idx, n_nodes, n_edges)

    图的数据容器。

    边信息以COO格式存储。

    参数：
        - **src_idx** (Tensor) - shape为 :math:`(N\_EDGES)` 的int类型Tensor，表示COO边矩阵的源节点索引。
        - **dst_idx** (Tensor) - shape为 :math:`(N\_EDGES)` 的int类型Tensor，表示COO边矩阵的目标节点索引。
        - **n_nodes** (int) - 图中节点数量。
        - **n_edges** (int) - 图中边数量。

    .. py:method:: mindspore_gl.GraphField.get_graph

        获取图。

        返回:
            List，Tensor的列表，被应用于构造函数。