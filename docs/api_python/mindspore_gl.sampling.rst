mindspore_gl.sampling
=====================

.. py:class:: mindspore_gl.sampling

    图数据的采样API。

.. py:function:: mindspore_gl.sampling.k_hop_subgraph(node_idx, num_hops, adj_coo, node_count, relabel_nodes=False, flow='source_to_target')

    MindHomoGraph上的K跳采样。

    参数：
        - **node_idx** (int, list, tuple or numpy.ndarray) - 围绕 `node_idx` 采样子图。
        - **num_hops** (int) - 在子图上采样 `num_hops` 跳。
        - **ad_coo** (numpy.ndarray) - 输入图的邻接矩阵。
        - **node_count** (int) - 节点数。
        - **relabel_nodes** (bool) - 节点索引是否需要重新标签。默认值：False。
        - **flow** (str) - 访问方向。默认值：source_to_target。

    返回：
        res(dict)，有4个键“subset”、“ad_coo”、“inv”、“edge_mask”，其中，

        - **subset** (numpy.array) - 采样K跳的子图节点idx。
        - **ad_coo** (numpy.array) - 采样K跳的子图邻接矩阵。
        - **inv** (list) - 从 `node_idx` 中的节点索引到其新位置的映射。
        - **edge_mask** (numpy.array) - 边掩码，指示保留哪些边。

    异常：
        - **TypeError** - 如果 `num_hops` 或 `num_hops` 不是正整数。
        - **TypeError** - 如果 `num_hops` 不是bool。
        - **ValueError** - 如果 `flow` 不是source_to_target或target_to_source。

.. py:function:: mindspore_gl.sampling.negative_sample(positive, node, num_neg_samples, mode='undirected', re='more')

    输入所有正样本边缘集，并指定负样本长度。
    然后返回相同长度的负样本边缘集，并且不会重复正样本。
    可以选择考虑自循环、有向图或无向图操作。

    参数：
        - **positive** (list or array) - 所有正样本边，shape为 :math:`(col_len,row_len)`
        - **node** (int) - 节点数。
        - **num_neg_samples** (int) - 负样本长度。
        - **mode** (str) - 运算矩阵的类型。默认值：undirected。
        - **re** (str) - 输入数据类型。默认值：more。

    返回：
        数组，负采样边集，shape为 :math:`(num_neg_samples, 2)`

    异常：
        - **TypeError** - 如果 `positive` 不是List或numpy.ndarry。
        - **TypeError** - 如果 `node` 不是正整数。
        - **TypeError** - 如果 `re` 不在more或其他中。
        - **ValueError** - 如果 `mode` 不是bipartite、undirected或other。

.. py:function:: mindspore_gl.sampling.random_walk_unbias_on_homo(homo_graph: mindspore_gl.graph.graph.MindHomoGraph, seeds: numpy.ndarray, walk_length: int)

    同构图上的随机游走

    参数：
        - **homo_graph** (MindHomoGraph) - 采样的源图。
        - **seeds** (np.ndarray) - 用于采样的随机种子。
        - **walk_length** (int) - 采样路径长度。

    异常：
        - **TypeError** - 如果 `walk_length` 不是正整数。
        - **TypeError** - 如果 `seeds` 不是numpy.int32。

    返回：
        数组，示例节点 :math:`(len(seeds), walk_length)`

.. py:function:: mindspore_gl.sampling.sage_sampler_on_homo(homo_graph: mindspore_gl.graph.graph.MindHomoGraph, seeds: <built-in function array>, neighbor_nums: List[int])

    MindHomoGraph上的GraphSage采样。

    参数：
        - **homo_graph** (MindHomoGraph) - 输入图。
        - **seeds** (numpy.array) - 邻居采样的起始节点。
        - **neighbor_nums** (List) - 每跳的邻居数量。

    返回：
        layered_edges_{idx}(numpy.array)，第idx跳时采样的边数组。

        layered_eids_{idx}(numpy.array)，第idx跳时采样的边上点的ID。

        all_nodes，所有节点的ID。

        seeds_idx，种子的ID。

    异常：
        - **TypeError** - 如果 `homo_graph` 不是MindHomoGraph类。
        - **TypeError** - 如果 `seeds` 不是numpy.array。
        - **TypeError** - 如果 `neighbor_nums` 不是List。
