mindspore_gl.graph.MindHomoGraph
================================

.. py:class:: mindspore_gl.graph.MindHomoGraph

    构建同构图。

    .. py:method:: mindspore_gl.graph.MindHomoGraph.adj_coo
        :property:

        coo调整矩阵。

        返回：
            numpy.ndarray，coo图。

    .. py:method:: mindspore_gl.graph.MindHomoGraph.adj_csr
        :property:

        csr调整矩阵。

        返回：
            mindspore_gl.graph.csr_adj，csr图。

    .. py:method:: mindspore_gl.graph.MindHomoGraph.batch_meta
        :property:

        如果图形被批处理。

        返回：
            mindspore_gl.graph.BatchMeta，batch meta信息。

    .. py:method:: mindspore_gl.graph.MindHomoGraph.degree(node)

        Lazy计算查询节点的度。

        参数：
            - **node** (int) - 节点ID。

        返回：
            int，节点的度。

    .. py:method:: mindspore_gl.graph.MindHomoGraph.edge_count
        :property:

        图的边数。

        返回：
            int，边数。

    .. py:method:: mindspore_gl.graph.MindHomoGraph.is_batched
        :property:

        如果图被批处理。

        返回：
            bool，图被批处理。

    .. py:method:: mindspore_gl.graph.MindHomoGraph.neighbors(node)

        Lazy计算查询。

        参数：
            - **node** (int) - 节点ID。

        返回：
            numpy.ndarray，采样的节点。

    .. py:method:: mindspore_gl.graph.MindHomoGraph.node_count
        :property:

        图的节点数。

        返回：
            int，节点号。

    .. py:method:: mindspore_gl.graph.MindHomoGraph.set_topo(adj_csr: np.ndarray, node_dict, edge_ids: np.ndarray)

        初始化CSR图。

        参数：
            - **adj_csr** (mindspore_gl.graph.csr_adj) - 图的邻接矩阵，csr格式。
            - **node_dict** (dict) - 节点id字典。
            - **edge_ids** (numpy.ndarray) - 边数组。

    .. py:method:: mindspore_gl.graph.MindHomoGraph.set_topo_coo(adj_coo, node_dict=None, edge_ids: np.ndarray = None)

        初始化COO图。

        参数：
            - **adj_coo** (numpy.ndarray) - 图的邻接矩阵，coo格式。
            - **node_dict** (dict) - 节点id字典。默认值：None。
            - **edge_ids** (numpy.ndarray) - 边数组。默认值：None。
