mindspore_gl.graph.get_laplacian
================================

.. py:function:: mindspore_gl.graph.get_laplacian(edge_index, edge_weight, normalization, num_nodes)

    获得laplacian矩阵。

    输入：
         - **edge_index** (Tensor) - 边索引。shape为 :math:`(2, N\_e)` 其中 :math:`N\_e` 是边的数量。
         - **edge_weight** (Tensor) - 边权重。shape为 :math:`(N\_e)` 其中 :math:`N\_e` 是边的数量。
         - **normalization** (str) - 归一化方法。

           1. :obj:`None` ：无
              :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

           2. :obj:`"sym"` ：对称归一化
              :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}`

           3. :obj:`"rw"` ：随机游走归一化
              :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

         - **num_nodes** (int) - 节点数。

    返回：
         - **edge_index** (Tensor) - 标准化的边索引。
         - **edge_weight** (Tensor) - 归一化边权重。

    异常：
        - **ValueError** - 如果 `normalization` 不是None、'sym'或'rw'。
