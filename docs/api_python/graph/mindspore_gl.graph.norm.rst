mindspore_gl.graph.norm
=======================

.. py:function:: mindspore_gl.graph.norm(edge_index, num_nodes, edge_weight=None, normalization='sym', lambda_max=None, batch=None)

    图laplacian归一化。

    参数：
        - **edge_index** (Tensor) - 边索引。Shape为 :math:`(2, N\_e)` 其中 :math:`N\_e` 是边的数量。
        - **num_nodes** (int) - 节点数。
        - **edge_weight** (Tensor) - 边权重。Shape为 :math:`(N\_e)` 其中 :math:`N\_e` 是边的数量。默认值：None。
        - **normalization** (str) - 归一化方法。默认值： 'sym'。

          :math:`(L)` 为归一化的矩阵， :math:`(D)` 为度矩阵， :math:`(A)` 为邻接矩阵， :math:`(I)` 为单元矩阵。

          1. `None`：无
             :math:`\mathbf{L}=\mathbf{D}-\mathbf{A}`

          2. `'sym'`：对称归一化
             :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}`

          3. `'rw'`：随机游走归一化
             :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

        - **lambda_max** (int, float) - 图的Lambda值。默认值：None。
        - **batch** (Tensor) - 批处理向量。默认值：None。

    返回：
         - **edge_index** (Tensor) - 标准化边索引。
         - **edge_weight** (Tensor) - 归一化边权重。

    异常：
        - **ValueError** - 如果 `normalization` 不是None、'sym'或'rw'。
