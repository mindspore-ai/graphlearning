mindspore_gl.graph.add_self_loop
================================

.. py:function:: mindspore_gl.graph.add_self_loop(edge_index, edge_weight, node, fill_value, mode='dense')

    从输入COO矩阵中添加自循环。
    可以选择对dense矩阵或COO格式的矩阵进行操作。

    参数：
        - **edge_index** (Tensor) - 边索引。shape为 :math:`(2, N\_e)` 其中 :math:`N\_e` 是边的数量。
        - **edge_weight** (Tensor) - 边权重。shape为 :math:`(N\_e)` 其中 :math:`N\_e` 是边的数量。
        - **node** (int) - 节点数。
        - **fill_value** (Tensor) - 自循环值。
        - **mode** (str, 可选) - 操作矩阵的类型。支持的类型为 ``'dense'`` 和 ``'coo'``。默认值：``'dense'``。

    返回：
        如果 `mode` 等于 ``'dense'``：

        - **new_adj** (Tensor) - 添加对角矩阵后的对象。

        如果 `mode` 等于 ``'coo'``：

        - **edge_index** (Tensor) - 新的边索引。
        - **edge_weight** (Tensor) - 新的边权重。

    异常：
        - **ValueError** - 如果 `mode` 不是 ``'coo'`` 或 ``'dense'``。
        - **TypeError** - 如果 `node` 不是正整数。
        - **ValueError** - 如果 `fill_value` 长度不等于 `node` 。
