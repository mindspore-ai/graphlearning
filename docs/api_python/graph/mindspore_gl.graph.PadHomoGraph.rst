mindspore_gl.graph.PadHomoGraph
===============================

.. py:class:: mindspore_gl.graph.PadHomoGraph(n_node=None, mode=PadMode.AUTO, n_edge=None)

    Pad MindHomoGraph，通过在这些节点之间添加额外的节点和边来填充图形。简言之，PadHomoGraph(graph1) = BatchHomoGraph(graph1, fake_graph)
    虚构图中的节点计数和边计数由用户特定的参数决定。

    参数：
        - **n_node** (Union(int, None)) - 目标图的节点计数。默认值：None。
        - **n_edge** (Union(int, None)) - 目标图的边计数。默认值：None。
        - **mode** (PadMode) - Pad模式，如果选择PadMode.CONST，虚构图将具有n_node数量的节点和n_edge数量的边。如果为PadMode.AUTO，虚构图的node_count和edge_count是根据输入图的大小通过 :math:`n\_node = 2^ceil(log2(input\_graph.node\_count)), n_edge = 2^ceil(log2(input\_graph.edge\_count))` 计算的。默认值：PadMode.AUTO。

    输入：
        - **graph** (MindHomoGraph) - 输入图。

    输出：
        MindHomoGraph，填充图。
