mindspore_gl.BatchedGraphField
===============================

.. py:class:: mindspore_gl.BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx, edge_subgraph_idx, graph_mask)

    批次图的数据容器。

    边信息以COO格式存储。

    参数：
        - **src_idx** (Tensor) - shape为 :math:`(N\_EDGES)` 的int类型Tensor，表示COO边矩阵的源节点索引。
        - **dst_idx** (Tensor) - shape为 :math:`(N\_EDGES)` 的int类型Tensor，表示COO边矩阵的目标节点索引。
        - **n_nodes** (int) - 图中节点数量。
        - **n_edges** (int) - 图中边数量。
        - **ver_subgraph_idx** (Tensor) - shape为 :math:`(N\_NODES)` 的int类型Tensor，指示每个节点属于哪个子图。
        - **edge_subgraph_idx** (Tensor) - shape为 :math:`(N\_NODES,)` 的int类型Tensor，指示每个边属于哪个子图。
        - **graph_mask** (Tensor) - shape为 :math:`(N\_GRAPHS,)` 的int类型Tensor，表示子图是否存在。

    .. py:method:: mindspore_gl.BatchedGraphField.get_batched_graph

        获取批次图。

        返回:
            List，Tensor的列表，被应用于构造函数。