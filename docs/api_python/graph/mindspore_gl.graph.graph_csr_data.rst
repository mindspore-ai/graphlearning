mindspore_gl.graph.graph_csr_data
======================================

.. py:function:: graph_csr_data(src_idx, dst_idx, n_nodes, n_edges, node_feat=None, node_label=None, train_mask=None, val_mask=None, test_mask=None, rerank=False)

    将COO类型的整图转为CSR类型。

    参数：
        - **src_idx** (Union[Tensor, numpy.ndarray]) - shape为 :math:`(N\_EDGES)` 的int类型Tensor，表示COO边矩阵的源节点索引。
        - **dst_idx** (Union[Tensor, numpy.ndarray]) - shape为 :math:`(N\_EDGES)` 的int类型Tensor，表示COO边矩阵的目标节点索引。
        - **n_nodes** (int) - 图中节点数量。
        - **n_edges** (int) - 图中边数量。
        - **node_feat** (Union[Tensor, numpy.ndarray, 可选]) - 节点特征。
        - **node_label** (Union[Tensor, numpy.ndarray, 可选]) - 节点标签。
        - **train_mask** (Union[Tensor, numpy.ndarray, 可选]) - 训练索引的掩码。
        - **val_mask** (Union[Tensor, numpy.ndarray, 可选]) - 验证索引的掩码。
        - **test_mask** (Union[Tensor, numpy.ndarray, 可选]) - 测试索引的掩码。
        - **rerank** (bool, 可选) - 是否对节点特征、标签、掩码进行重排序。默认值：False。

    返回：
        - **csr_g** (tuple) - CSR图的信息，它包含CSR图的indices，CSR图的indptr，CSR图的节点数、CSR图的边数、CSR图的预存的反向indices、CSR图的预存储反向indptr。
        - **in_deg** - 每个节点的入度。
        - **out_deg** - 每个节点的出度。
        - **node_feat** (Union[Tensor, numpy.ndarray, 可选]) - 重排序的节点特征。
        - **node_label** (Union[Tensor, numpy.ndarray, 可选]) - 重排序的节点标签。
        - **train_mask** (Union[Tensor, numpy.ndarray, 可选]) - 重排序的训练索引的掩码。
        - **val_mask** (Union[Tensor, numpy.ndarray, 可选]) - 重排序的验证索引的掩码。
        - **test_mask** (Union[Tensor, numpy.ndarray, 可选]) - 重排序的测试索引的掩码。
