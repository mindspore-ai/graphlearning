mindspore_gl.graph.remove_self_loop
===================================

.. py:function:: mindspore_gl.graph.remove_self_loop(adj, mode='dense')

    从输入矩阵对象中删除对角矩阵，可以选择对dense矩阵或COO格式的矩阵进行操作。

    参数：
        - **adj** (scipy.sparse.coo) - 目标矩阵。
        - **mode** (str, 可选) - 操作矩阵的类型。支持的图类型为'coo'和'dense'。默认值：'dense'。

    返回：
        移除对角矩阵后的对象。
        如果 `mode` 为dense，返回Tensor类型；如果 `mode` 为coo返回spy.sparse.coo类型。
