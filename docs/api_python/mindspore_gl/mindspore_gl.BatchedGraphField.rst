mindspore_gl.BatchedGraphField
===============================

.. py:class:: mindspore_gl.BatchedGraphField(src_idx=None, dst_idx=None, n_nodes=None, n_edges=None, ver_subgraph_idx=None, edge_subgraph_idx=None, graph_mask=None, indices=None, indptr=None, indices_backward=None, indptr_backward=None, csr=False)

    批次图的数据容器。

    边信息以COO格式存储。

    参数：
        - **src_idx** (Tensor, 可选) - shape为 :math:`(N\_EDGES)` 的int类型Tensor，表示COO边矩阵的源节点索引。默认值：``None``。
        - **dst_idx** (Tensor, 可选) - shape为 :math:`(N\_EDGES)` 的int类型Tensor，表示COO边矩阵的目标节点索引。默认值：``None``。
        - **n_nodes** (int, 可选) - 图中节点数量。默认值：``None``。
        - **n_edges** (int, 可选) - 图中边数量。默认值：``None``。
        - **ver_subgraph_idx** (Tensor, 可选) - shape为 :math:`(N\_NODES)` 的int类型Tensor，指示每个节点属于哪个子图。默认值：``None``。
        - **edge_subgraph_idx** (Tensor, 可选) - shape为 :math:`(N\_NODES,)` 的int类型Tensor，指示每个边属于哪个子图。默认值：``None``。
        - **graph_mask** (Tensor, 可选) - shape为 :math:`(N\_GRAPHS,)` 的int类型Tensor，表示子图是否存在。默认值：``None``。
        - **indices** (Tensor, 可选) - shape为 :math:`(N\_EDGES)` 的int类型Tensor，CSR矩阵中的indices。默认值：``None``。
        - **indptr** (Tensor, 可选) - shape为 :math:`(N\_NODES,)` 的int类型Tensor，CSR矩阵中的indptr。默认值：``None``。
        - **indices_backward** (Tensor, 可选) - shape为 :math:`(N\_EDGES)` 的int类型Tensor，CSR矩阵中的预存的indices反向。默认值：``None``。
        - **indptr_backward** (Tensor, 可选) - shape为 :math:`(N\_NODES,)` 的int类型Tensor，CSR矩阵中的预存的indptr反向。默认值：``None``。
        - **csr** (bool, 可选) - 是否为CSR类型。默认值：``False``。

    .. py:method:: mindspore_gl.BatchedGraphField.get_batched_graph

        获取批次图。

        返回：
            List，Tensor的列表，被应用于构造函数。