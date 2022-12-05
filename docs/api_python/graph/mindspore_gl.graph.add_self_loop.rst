mindspore_gl.graph.add_self_loop
================================

.. py:function:: mindspore_gl.graph.add_self_loop(adj, node, fill_value, mode='dense')

    功能：
    从输入coo矩阵中添加自循环。
    可以选择对dense矩阵或coo格式的矩阵进行操作。

    参数：
        - **adj** (Tensor) - COO矩阵。
        - **node** (int) - 节点数。
        - **fill_value** (Tensor) - 自循环值。
        - **mode** (str) - 操作矩阵的类型。默认值：dense。

    返回：
        - **new_adj** (Tensor) - 添加对角矩阵后的对象。
          'dense'返回dense Tensor类型。
          'coo'返回coo Tensor类型。

    异常：
        - **ValueError** - 如果 `mode` 不是coo或dense格式。
        - **TypeError** - 如果 `node` 不是正整数。
