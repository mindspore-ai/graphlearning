mindspore_gl.BatchedGraph
=========================

.. py:class:: mindspore_gl.BatchedGraph

    批次图类。

    在 `GNNCell` 类的构造函数中需要被注释的类。`construct` 函数中的最后一个参数将被解析成 `mindspore_gl.BatchedGraph` 批次图类。

    .. py:method:: mindspore_gl.BatchedGraph.avg_edges(edge_feat)

        聚合边特征，通过聚合函数“平均”来生成图级表示。

        节点特征为 :math:`(N\_EDGES, F)` ， `max_edges` 将会根据 `edge_subgraph_idx` 进行节点特征聚合操作。
        输出的Tensor维度为 :math:`(N\_GRAPHS, F)` ， :math:`F` 为节点特征维度。

        参数:
            - **edge_feat** (Tensor) - 节点特征，shape为 :math:`(N\_EDGES, F)` ，`F` 是特征维度。

        返回：
            Tensor，shape为 :math:`(N\_GRAPHS, F)` ， :math:`F` 是节点特征维度。

        异常：
            - **TypeError** - 如果 `node_feat` 不是Tensor。

    .. py:method:: mindspore_gl.BatchedGraph.avg_nodes(node_feat)

        聚合邻居的节点特征，通过聚合函数“平均”来生成图级表示。

        节点特征为 :math:`(N\_NODES, F)` ， `max_nodes` 将会根据 `ver_subgraph_idx` 进行节点特征聚合操作。
        输出的Tensor维度为 :math:`(N\_GRAPHS, F)` ， :math:`F` 为节点特征维度。

        参数:
            - **node_feat** (Tensor) - 节点特征，shape为 :math:`(N\_NODES, F)` ，`F` 是特征维度。

        返回：
            Tensor，shape为 :math:`(N\_GRAPHS, F)` ， :math:`F` 是节点特征维度。

        异常：
            - **TypeError** - 如果 `node_feat` 不是Tensor。

    .. py:method:: mindspore_gl.BatchedGraph.broadcast_edges(graph_feat)

        将图级特征广播到边级特征表示。

        参数:
            - **graph_feat** (Tensor) - 节点特征，shape为 :math:`(N\_NODES, F)` ，`F` 是特征维度。

        返回：
            Tensor，shape为 :math:`(N\_EDGES, F)` ， :math:`F` 是特征维度。

        异常：
            - **TypeError** - 如果 `graph_feat` 不是Tensor。

    .. py:method:: mindspore_gl.BatchedGraph.broadcast_nodes(graph_feat)

        将图级特征广播到节点级特征表示。

        参数:
            - **node_feat** (Tensor) - 节点特征，shape为 :math:`(N\_NODES, F)` ，`F` 是特征维度。

        返回：
            Tensor，shape为 :math:`(N\_NODES, F)` ， :math:`F` 是特征维度。

        异常：
            - **TypeError** - 如果 `graph_feat` 不是Tensor。

    .. py:method:: mindspore_gl.BatchedGraph.edge_mask

        获取padding之后的边的掩码。

        边掩码根据 `mindspore_gl.BatchedGraph.graph_mask` 和 `mindspore_gl.BatchedGraph.edge_subgraph_idx` 计算出来。在掩码中，1表示边存在，0表示边是通过padding生成的。

        返回：
            Tensor，shape为 :math:`(N\_EDGES,)` 。在Tensor中，1表示节点存在，0表示节点是通过padding生成的。

    .. py:method:: mindspore_gl.BatchedGraph.edge_subgraph_idx
        :property:

        指示边属于哪个子图。

        返回：
            Tensor，shape为 :math:`(N\_EDGES,)` 。

    .. py:method:: mindspore_gl.BatchedGraph.graph_mask
        :property:

        指示哪个子图是真实存在的。

        返回：
            Tensor，shape为 :math:`(N\_GRAPHS,)` 。

    .. py:method:: mindspore_gl.BatchedGraph.max_edges(edge_feat)

        聚合边特征，通过聚合函数“最大值”来生成图级表示。

        节点特征为 :math:`(N\_EDGES, F)` ， `max_edges` 将会根据 `edge_subgraph_idx` 进行节点特征聚合操作。
        输出的Tensor维度为 :math:`(N\_GRAPHS, F)` ， :math:`F` 为节点特征维度。

        参数:
            - **edge_feat** (Tensor) - 节点特征，shape为 :math:`(N\_EDGES, F)` ，`F` 是特征维度。

        返回：
            Tensor，shape为 :math:`(N\_GRAPHS, F)` ， :math:`F` 是节点特征维度。

        异常：
            - **TypeError** - 如果 `node_feat` 不是Tensor。

    .. py:method:: mindspore_gl.BatchedGraph.max_nodes(node_feat)

        聚合邻居的节点特征，通过聚合函数“最大值”来生成图级表示。

        节点特征为 :math:`(N\_NODES, F)` ， `max_nodes` 将会根据 `ver_subgraph_idx` 进行节点特征聚合操作。
        输出的Tensor维度为 :math:`(N\_GRAPHS, F)` ， :math:`F` 为节点特征维度。

        参数:
            - **node_feat** (Tensor) - 节点特征，shape为 :math:`(N\_NODES, F)` ，`F` 是特征维度。

        返回：
            Tensor，shape为 :math:`(N\_GRAPHS, F)` ， :math:`F` 是节点特征维度。

        异常：
            - **TypeError** - 如果 `node_feat` 不是Tensor。

    .. py:method:: mindspore_gl.BatchedGraph.n_graphs
        :property:

        表示批次图由多少个子图组成。

        返回：
            int，图的数量。

    .. py:method:: mindspore_gl.BatchedGraph.node_mask

        获取padding之后的节点的掩码。在掩码中，1表示节点存在，0表示节点是通过padding生成的。

        节点掩码根据 `mindspore_gl.BatchedGraph.graph_mask` 和 `mindspore_gl.BatchedGraph.ver_subgraph_idx` 计算出来。

        返回：
            Tensor，shape为 :math:`(N\_NODES,)` 。在Tensor中，1表示节点存在，0表示节点是通过padding生成的。

    .. py:method:: mindspore_gl.BatchedGraph.num_of_edges

        获取批次图中每个子图的边数量。

        .. note::
            填充操作后，将创建一个不存在的子图，并且创建的所有不存在的边都属于该子图。
            如果要清除它，则需要手动将它与 `graph_mask` 相乘。

        返回：
            Tensor，shape为 :math:`(N\_GRAPHS, 1)` ，表示每个子图有多少边。

    .. py:method:: mindspore_gl.BatchedGraph.num_of_nodes

        获取批次图中每个子图的节点数量。

        .. note::
            填充操作后，将创建一个不存在的子图，并且创建的所有不存在的节点都属于该子图。
            如果要清除它，则需要手动将它与 `graph_mask` 相乘。

        返回：
            Tensor，shape为 :math:`(N\_GRAPHS, 1)` ，表示每个子图有多少节点。

    .. py:method:: mindspore_gl.BatchedGraph.softmax_edges(edge_feat)

        对边特征执行图的softmax。

        针对每个边 :math:`v\in\mathcal{V}` 和它的特征 :math:`x_v` ，计算归一化方法如下:

        .. math::
            z_v = \frac{\exp(x_v)}{\sum_{u\in\mathcal{V}}\exp(x_u)}

        每个子图独立计算softmax。
        结果Tensor具有与原始边特征相同的shape。

        参数:
            - **edge_feat** (Tensor) - 边特征的Tensor，shape为 :math:`(N\_EDGES, F)` ，`F` 是特征维度。

        返回：
            Tensor，shape为 :math:`(N\_EDGES, F)` ， :math:`F` 是节点特征维度。

        异常：
            - **TypeError** - 如果 `edge_feat` 不是Tensor。

    .. py:method:: mindspore_gl.BatchedGraph.softmax_nodes(node_feat)

        对节点特征执行图的softmax。

        针对每个节点 :math:`v\in\mathcal{V}` 和它的特征 :math:`x_v` ，计算归一化方法如下:

        .. math::
            z_v = \frac{\exp(x_v)}{\sum_{u\in\mathcal{V}}\exp(x_u)}

        每个子图独立计算softmax。
        结果Tensor具有与原始节点特征相同的shape。

        参数:
            - **node_feat** (Tensor) - 节点特征，shape为 :math:`(N\_NODES, F)` ，`F` 是特征维度。

        返回：
            Tensor，shape为 :math:`(N\_NODES, F)` ， :math:`F` 是节点特征维度。

        异常：
            - **TypeError** - 如果 `node_feat` 不是Tensor。

    .. py:method:: mindspore_gl.BatchedGraph.sum_edges(edge_feat)

        聚合边特征，通过聚合函数“求和”来生成图级表示。

        边特征为 :math:`(N\_EDGES, F)` ， `sum_edges` 将会根据 `edge_subgraph_idx` 进行节点特征聚合操作。
        输出的Tensor维度为 :math:`(N\_GRAPHS, F)` ， :math:`F` 为节点特征维度。

        参数:
            - **edge_feat** (Tensor) - 节点特征，shape为 :math:`(N\_EDGES, F)` ，`F` 是特征维度。

        返回：
            Tensor，shape为 :math:`(N\_GRAPHS, F)` ， :math:`F` 是节点特征维度。

        异常：
            - **TypeError** - 如果 `edge_feat` 不是Tensor。

    .. py:method:: mindspore_gl.BatchedGraph.sum_nodes(node_feat)

        聚合邻居的节点特征，通过聚合函数“求和”来生成图级表示。

        节点特征为 :math:`(N\_NODES, F)` ， `sum_nodes` 将会根据 `ver_subgraph_idx` 进行节点特征聚合操作。
        输出的Tensor维度为 :math:`(N\_GRAPHS, F)` ， :math:`F` 为节点特征维度。

        参数:
            - **node_feat** (Tensor) - 节点特征，shape为 :math:`(N\_NODES, F)` ，`F` 是特征维度。

        返回：
            Tensor，shape为 :math:`(N\_GRAPHS, F)` ， :math:`F` 是节点特征维度。

        异常：
            - **TypeError** - 如果 `node_feat` 不是Tensor。

    .. py:method:: mindspore_gl.BatchedGraph.ver_subgraph_idx
        :property:

        指示节点属于哪个子图。

        返回：
            Tensor，shape为 :math:`(N)` ， :math:`N` 为图中节点个数。
