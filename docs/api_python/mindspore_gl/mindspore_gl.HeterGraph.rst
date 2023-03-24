mindspore_gl.HeterGraph
=======================

.. py:class:: mindspore_gl.HeterGraph

    异构图类。

    在 `GNNCell` 类的构造函数中需要被注释的类。`construct` 函数中的最后一个参数将被解析成 `mindspore_gl.HeterGraph` 异构图类。

    .. py:method:: mindspore_gl.Graph.dst_idx
        :property:

        一个具有shape为 :math:`(N\_EDGES)` 的Tensor，表示COO边矩阵的目标节点索引。

        返回：
            List[Tensor]，目标顶点列表。

    .. py:method:: mindspore_gl.Graph.get_homo_graph(etype)

        获取特定的etype的节点、边。

        参数：
            - **etype** (int) - 边类型。

        返回:
            List[Tensor]，同构图。

    .. py:method:: mindspore_gl.Graph.n_edges
        :property:

        图的边数。

        返回:
            List[int]，图的边数的列表。

    .. py:method:: mindspore_gl.Graph.n_nodes
        :property:

        图的节点数。

        返回:
            List[int]，图的节点数的列表。

    .. py:method:: mindspore_gl.Graph.src_idx
        :property:

        一个具有shape为 :math:`(N\_EDGES)` 的Tensor，表示COO边矩阵的源节点索引。

        返回：
            List[Tensor]，源顶点列表。
