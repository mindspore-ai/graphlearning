mindspore_gl.graph.BatchMeta
============================

.. py:class:: mindspore_gl.graph.BatchMeta(graph_nodes, graph_edges)

    BatchMeta，批处理图形的元信息。

    参数：
        - **graph_nodes** (numpy.array) - 批处理图中图的累积节点和（第一个元素为0）。
        - **graph_edges** (numpy.array) - 批处理图中图的累积边缘和（第一个元素为0）。

    .. py:method:: mindspore_gl.graph.BatchMeta.edge_map_idx
        :property:

        边的索引列表。

        返回：
            numpy.array，数组，指示每个边的图索引。

    .. py:method:: mindspore_gl.graph.BatchMeta.graph_count
        :property:

        图的数量。

        返回：
            int，此批处理图中的总图计数。

    .. py:method:: mindspore_gl.graph.BatchMeta.graph_edges
        :property:

        图的边。

        返回：
            numpy.array，批处理图中图的累积边和（第一个元素为0）。

    .. py:method:: mindspore_gl.graph.BatchMeta.graph_nodes
        :property:

        图的节点。

        返回：
            numpy.array，批处理图中图的累积节点和（第一个元素为0）。

    .. py:method:: mindspore_gl.graph.BatchMeta.node_map_idx
        :property:

        节点的索引列表。

        返回：
            numpy.array，数组，指示每个节点的图索引。
