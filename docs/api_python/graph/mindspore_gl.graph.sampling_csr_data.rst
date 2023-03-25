mindspore_gl.graph.batch_graph_csr_data
============================================

.. py:function:: batch_graph_csr_data(src_idx, dst_idx, n_nodes, n_edges, seeds_idx=None, node_feat=None, rerank=False)

    将COO类型的采样图转为CSR类型。

    参数：
        - **src_idx** (Union[Tensor, numpy.ndarray]) - shape为 :math:`(N\_EDGES)` 的int类型Tensor，表示COO边矩阵的源节点索引。
        - **dst_idx** (Union[Tensor, numpy.ndarray]) - shape为 :math:`(N\_EDGES)` 的int类型Tensor，表示COO边矩阵的目标节点索引。
        - **n_nodes** (int) - 图中节点数量。
        - **n_edges** (int) - 图中边数量。
        - **seeds_idx** (numpy.ndarray) - 初始邻居采样节点。
        - **node_feat** (Union[Tensor, numpy.ndarray, 可选]) - 节点特征。
        - **rerank** (bool, 可选) - 是否对节点特征、标签、掩码进行重排序。默认值：False。

    返回：
        - **csr_g** (tuple) - CSR图的信息，它包含CSR图的indices，CSR图的indptr，CSR图的节点数、CSR图的边数、CSR图的预存的反向indices、CSR图的预存储反向indptr。
        - **seeds_idx** (numpy.ndarray) - 重排序的初始采样节点。
        - **node_feat** (numpy.ndarray) - 重排序的节点特征。
