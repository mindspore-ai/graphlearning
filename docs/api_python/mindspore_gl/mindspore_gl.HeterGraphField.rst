mindspore_gl.HeterGraphField
============================

.. py:class:: mindspore_gl.HeterGraphField(src_idx, dst_idx, n_nodes, n_edges)

    异构图的数据容器。

    边信息以COO格式存储。

    参数：
        - **src_idx** (List[Tensor]) - 包含Tensor的list，Tensor的shape为 :math:`(N\_EDGES)` 的int类型Tensor，表示COO边矩阵的源节点索引。
        - **dst_idx** (List[Tensor]) - 包含Tensor的list，Tensor的shape为 :math:`(N\_EDGES)` 的int类型Tensor，表示COO边矩阵的目标节点索引。
        - **n_nodes** (List[int]) - 由int组成的list，表示图中节点数量。
        - **n_edges** (List[int]) - 由int组成的list，表示图中边数量。

    .. py:method:: mindspore_gl.HeterGraphField.get_heter_graph

        获取批次图。

        返回:
            List，Tensor的列表，被应用于构造函数。
