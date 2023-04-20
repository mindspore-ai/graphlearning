mindspore_gl.Graph
==================

.. py:class:: mindspore_gl.Graph

    图类。

    在 `GNNCell` 类的构造函数中需要被注释的类。 `construct` 函数中的最后一个参数将被解析成 `mindspore_gl.Graph` 整图类。

    .. py:method:: mindspore_gl.Graph.adj_to_dense

        获取图的稠密相邻矩阵。

        .. note::
            由于系统限制，目前只支持COO格式进行构建图，并生成Dense格式邻接矩阵。

        返回：
            Tensor，shape为 :math:`(N, N)` ， :math:`N` 是图的节点数。

    .. py:method:: mindspore_gl.Graph.avg(neigh_feat)

        聚合邻居的节点特征，通过聚合函数“平均”来生成节点级表示。

        参数：
            - **neigh_feat** (List[`SrcVertex` feature or `Edge` feature]) - `SrcVertex` 或 `Edge` 表示相邻节点或边特征的属性列表，shape为 :math:`(N, F)` 。
              `N` 是 `SrcVertex` 或 `Edge` 的个数， `F` 是 `SrcVertex` 或 `Edge` 的特征维度。

        返回：
            mindspore.Tensor，shape为 :math:`(N, F)` ，`N` 为节点数， `F` 为节点的特征维度。

        异常：
            - **TypeError** - 如果 `neigh_feat` 不是 `Edge` 或 `SrcVertex` 的列表。

    .. py:method:: mindspore_gl.Graph.dot(feat_x, feat_y)

        两个节点Tensor的点乘操作。

        参数：
            - **feat_x** (`SrcVertex` feature or `DstVertex` feature) - `SrcVertex` 或 `DstVertex` 表示的图节点特征，shape为 :math:`(N, F)` 。
              `N` 是图上节点数量， `F` 是节点的特征维度。
            - **feat_y** (`SrcVertex` feature or `DstVertex` feature) - `SrcVertex` 或 `DstVertex` 表示的图节点特征，shape为 :math:`(N, F)` 。
              `N` 是图上节点数量， `F` 是节点的特征维度。

        返回：
            mindspore.Tensor，shape为 :math:`(N, 1)` ， `N` 为节点数。

        异常：
            - **TypeError** - 如果 `feat_x` 不在 'mul' 操作的支持类型[Tensor、Number、List、Tuple]中。
            - **TypeError** - 如果 `feat_y` 不在 'mul' 操作的支持类型[Tensor、Number、List、Tuple]中。

    .. py:method:: mindspore_gl.Graph.dst_idx
        :property:

        一个具有shape为 :math:`(N\_EDGES)` 的Tensor，表示COO边矩阵的目标节点索引。

        返回：
            mindspore.Tensor，目标顶点列表。

    .. py:method:: mindspore_gl.Graph.dst_vertex
        :property:

        循环遍历获取目标顶点的 `innbs` 入度节点。

        返回：
            mindspore.Tensor，目标节点列表。

    .. py:method:: mindspore_gl.Graph.in_degree

        获取图形中每个节点的入度。

        返回：
            Tensor，shape为 :math:`(N, 1)` ，表示每个节点的入度， :math:`N` 是图的节点数。

    .. py:method:: mindspore_gl.Graph.max(neigh_feat)

        聚合邻居的节点特征，通过聚合函数“最大值”来生成节点级表示。

        参数：
            - **neigh_feat** (List[`SrcVertex` feature or `Edge` feature]) - `SrcVertex` 或 `Edge` 表示相邻节点或边特征的属性列表，shape为 :math:`(N, F)` 。
              `N` 是 `SrcVertex` 或 `Edge` 的个数， `F` 是 `SrcVertex` 或 `Edge` 的特征维度。

        返回：
            mindspore.Tensor，shape为 :math:`(N, F)` ，`N` 为节点数， `F` 为节点的特征维度。

        异常：
            - **TypeError** - 如果 `neigh_feat` 不是 `Edge` 或 `SrcVertex` 的列表。

    .. py:method:: mindspore_gl.Graph.min(neigh_feat)

        聚合邻居的节点特征，通过聚合函数“最小值”来生成节点级表示。

        参数：
            - **neigh_feat** (List[`SrcVertex` feature or `Edge` feature]) - `SrcVertex` 或 `Edge` 表示相邻节点或边特征的属性列表，shape为 :math:`(N, F)` 。
              `N` 是 `SrcVertex` 或 `Edge` 的个数， `F` 是 `SrcVertex` 或 `Edge` 的特征维度。

        返回：
            mindspore.Tensor，shape为 :math:`(N, F)` ，`N` 为节点数， `F` 为节点的特征维度。

        异常：
            - **TypeError** - 如果 `neigh_feat` 不是 `Edge` 或 `SrcVertex` 的列表。

    .. py:method:: mindspore_gl.Graph.n_edges
        :property:

        图的边数。

        返回：
            int，图的边数。

    .. py:method:: mindspore_gl.Graph.n_nodes
        :property:

        图的节点数。

        返回：
            int，图的节点数。

    .. py:method:: mindspore_gl.Graph.out_degree

        获取图形中每个节点的出度。

        返回：
            Tensor，shape为 :math:`(N, 1)` ，表示每个节点的出度， :math:`N` 是图的节点数。

    .. py:method:: mindspore_gl.Graph.set_dst_attr(feat_dict)

        在以顶点为中心的环境中设置目标顶点的属性。
        参数 `feat_dict` 的key是属性的名称，value是属性的数据。

        参数：
            - **feat_dict** (Dict) - key的类型为str，value的类型为Tensor，shape为 :math:`(N\_NODES, F)` ，其中 :math:`F` 是特征维度。

        异常：
            - **TypeError** - 如果 `feat_dict` 不是dict。

        返回：
            mindspore.Tensor，目标顶点的特征。

    .. py:method:: mindspore_gl.Graph.set_edge_attr(feat_dict)

        在以顶点为中心的环境中设置边的属性。
        参数 `feat_dict` 的key是属性的名称，value是属性的数据。

        参数：
            - **feat_dict** (Dict) - key的类型为str，value的类型为Tensor，shape为 :math:`(N\_NODES, F)` ，其中 :math:`F` 是特征维度。
              当特征维度为1时，推荐的边特征shape为 :math:`(N\_EDGES, 1)` 。

        异常：
            - **TypeError** - 如果 `feat_dict` 不是dict。

        返回：
            mindspore.Tensor，边的特征。

    .. py:method:: mindspore_gl.Graph.set_graph_attr(feat_dict)

        在以顶点为中心的环境中设置整图的属性。
        参数 `feat_dict` 的key是属性的名称，value是属性的数据。

        参数：
            - **feat_dict** (Dict) - key的类型为str，value的为整图的特征。

        异常：
            - **TypeError** - 如果 `feat_dict` 不是dict。

        返回：
            mindspore.Tensor，图的特征。

    .. py:method:: mindspore_gl.Graph.set_src_attr(feat_dict)

        在以顶点为中心的环境中设置源顶点的属性。
        参数 `feat_dict` 的key是属性的名称，value是属性的数据。

        参数：
            - **feat_dict** (Dict) - key的类型为str，value的类型为Tensor，shape为 :math:`(N\_NODES, F)` ，其中 :math:`F` 是特征维度。

        异常：
            - **TypeError** - 如果 `feat_dict` 不是dict。

        返回：
            mindspore.Tensor，源顶点的特征。

    .. py:method:: mindspore_gl.Graph.set_vertex_attr(feat_dict)

        在以顶点为中心的环境中为顶点设置属性。
        参数 `feat_dict` 的key是属性的名称，value是属性的数据。

        .. note::
            `set_vertex_attr` 的功能等价于 `set_src_attr` + `set_dst_attr`

        参数：
            - **feat_dict** (Dict) - key的类型为str，value的类型为Tensor，shape为 :math:`(N\_NODES, F)` ，其中 :math:`F` 是特征维度。

        异常：
            - **TypeError** - 如果 `feat_dict` 不是dict。

        返回：
            mindspore.Tensor，顶点的特征。

    .. py:method:: mindspore_gl.Graph.src_idx
        :property:

        一个具有shape为 :math:`(N\_EDGES)` 的Tensor，表示COO边矩阵的源节点索引。

        返回：
            mindspore.Tensor，源顶点列表。

    .. py:method:: mindspore_gl.Graph.src_vertex
        :property:

        循环遍历获取目标顶点的 `outnbs` 出度节点。

        返回：
            mindspore.Tensor，源节点列表。

    .. py:method:: mindspore_gl.Graph.sum(neigh_feat)

        聚合邻居的节点特征，通过聚合函数“求和”来生成节点级表示。

        参数：
            - **neigh_feat** (List[`SrcVertex` feature or `Edge` feature]) - `SrcVertex` 或 `Edge` 表示相邻节点或边特征的属性列表，shape为 :math:`(N, F)` 。
              `N` 是 `SrcVertex` 或 `Edge` 的个数， `F` 是 `SrcVertex` 或 `Edge` 的特征维度。

        返回：
            mindspore.Tensor，shape为 :math:`(N, F)` ，`N` 为节点数， `F` 为节点的特征维度。

        异常：
            - **TypeError** - 如果 `neigh_feat` 不是 `Edge` 或 `SrcVertex` 的列表。

    .. py:method:: mindspore_gl.Graph.topk_edges(node_feat, k, sortby=None)

        通过图上top-k个节点特征来表征图的特征。

        如果排序方式设置为无，则函数将独立对所有维度执行top-k。

        .. note::
            将按选定维度排序的值（如果 `sortby` 为 ``None``，则为所有维度）大于零。

            由于通过零值来对特征进行填充，其余特征可能会被零覆盖。

        参数：
            - **node_feat** (Tensor) - 节点特征，shape为 :math:`(N\_NODES, F)` ，`F` 是特征维度。
            - **k** (int) - top-k的节点个数。
            - **sortby** (int) - 根据哪个特征维度排序。如果为 ``None``，则所有特征都独立排序。默认值：``None``。

        返回：
            - **topk_output** (Tensor) - 特征Tensor的shape为 :math:`(B, K, F)` ，其中 :math:`B` 为输入图的批次大小，
              :math:`K` 为输入的 'k', :math:`F` 为特征维度。
            - **topk_indices** (Tensor) - top-k的输出索引，shape为 :math:`(B, K)` （当 `sortby` 为 ``None`` 时， :math:`(B, K, F)`），
              其中 :math:`B` 为输入图的批次大小， :math:`F` 为特征维度。

        异常：
            - **TypeError** - 如果 `node_feat` 不是Tensor。
            - **TypeError** - 如果 `k` 不是int。
            - **ValueError** - 如果 `sortby` 不是int。

    .. py:method:: mindspore_gl.Graph.topk_nodes(node_feat, k, sortby=None)

        通过图上top-k个节点特征来表征图的特征。

        如果排序方式设置为无，则函数将独立对所有维度执行top-k。

        .. note::
            将按选定维度排序的值（如果 `sortby` 为 ``None`` ，则为所有维度）大于零。

            由于通过零值来对特征进行填充，其余特征可能会被零覆盖。

        参数：
            - **node_feat** (Tensor) - 节点特征，shape为 :math:`(N\_NODES, F)` ，`F` 是特征维度。
            - **k** (int) - top-k的节点个数。
            - **sortby** (int) - 根据哪个特征维度排序。如果为 ``None``，则所有特征都独立排序。默认值：``None``。

        返回：
            - **topk_output** (Tensor) - 特征Tensor的shape为 :math:`(B, K, F)` ，其中 :math:`B` 为输入图的批次大小，
                :math:`K` 为输入的 'k', :math:`F` 为特征维度。
            - **topk_indices** (Tensor) - top-k的输出索引，shape为 :math:`(B, K)` （当 `sortby` 为 ``None`` 时， :math:`(B, K, F)`），
                其中 :math:`B` 为输入图的批次大小， :math:`F` 为特征维度。

        异常：
            - **TypeError** - 如果 `node_feat` 不是Tensor。
            - **TypeError** - 如果 `k` 不是int。
            - **ValueError** - 如果 `sortby` 不是int。
