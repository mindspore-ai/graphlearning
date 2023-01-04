mindspore_gl.dataset.Enzymes
============================

.. py:class:: mindspore_gl.dataset.Enzymes(root)

    ENZYMES数据集，用于读取和解析ENZYMES数据集的源数据集。

    有关ENZYMES数据集：

    ENZYMES是蛋白质三级结构数据集(Borgwardt等人，2005年)，由来自布伦达酶数据库（Schomburg等人，2004年）的600个酶组成。任务是将每个酶正确地分配给6个EC顶级类中的一个。

    数据：

    - 图: 600
    - 节点: 32.63
    - Edges: 62.14
    - 类的数量: 6

    下载地址： `ENZYMES <https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/ENZYMES.zip>`_ 。

    您可以将数据集文件组织到以下目录结构中，并通过 `preprocess` API读取。

    .. code-block::

        .
        ├── ENZYMES_A.txt
        ├── ENZYMES_graph_indicator.txt
        ├── ENZYMES_graph_labels.txt
        ├── ENZYMES_node_attributes.txt
        ├── ENZYMES_node_labels.txt
        └── README.txt

    参数：
        - **root** (str) - 包含enzymes_with_mask.npz的根目录的路径。

    异常：
        - **TypeError** - 如果 `root` 不是str。
        - **RuntimeError** - 如果 `root` 不包含数据文件。

    .. py:method:: mindspore_gl.dataset.Enzymes.graph_count
        :property:

        图的总数量。

        返回：
            int，图的数量。

    .. py:method:: mindspore_gl.dataset.Enzymes.graph_edges
        :property:

        累计图边数。

        返回：
            numpy.ndarray，累积边数组。

    .. py:method:: mindspore_gl.dataset.Enzymes.graph_feat(graph_idx)

        图上每个节点的特征。

        参数：
            - **graph_idx** (int) - 图索引。

        返回：
            numpy.ndarray，图的节点特征。

    .. py:method:: mindspore_gl.dataset.Enzymes.graph_label
        :property:

        图标签。

        返回：
            numpy.ndarray，图标签数组。

    .. py:method:: mindspore_gl.dataset.Enzymes.graph_nodes
        :property:

        累计图节点数。

        返回：
            numpy.ndarray，累计节点数组。

    .. py:method:: mindspore_gl.dataset.Enzymes.label_dim
        :property:

        标签种类。

        返回：
            int，标签种类。

    .. py:method:: mindspore_gl.dataset.Enzymes.max_num_node
        :property:

        单张图中最大的节点数量。

        返回：
            int，节点数中的最大数。

    .. py:method:: mindspore_gl.dataset.Enzymes.node_feat
        :property:

        节点特征。

        返回：
            numpy.ndarray，节点特征数组。

    .. py:method:: mindspore_gl.dataset.Enzymes.num_features
        :property:

        每个节点的特征数量。

        返回：
            int，特征大小的数量。

    .. py:method:: mindspore_gl.dataset.Enzymes.test_graphs
        :property:

        测试图ID。

        返回：
            numpy.ndarray，测试图ID数组。

    .. py:method:: mindspore_gl.dataset.Enzymes.test_mask
        :property:

        测试节点掩码。

        返回：
            numpy.ndarray，掩码数组。

    .. py:method:: mindspore_gl.dataset.Enzymes.train_graphs
        :property:

        训练图ID。

        返回：
            numpy.ndarray，训练图ID数组。

    .. py:method:: mindspore_gl.dataset.Enzymes.train_mask
        :property:

        训练节点掩码。

        返回：
            numpy.ndarray，掩码数组。

    .. py:method:: mindspore_gl.dataset.Enzymes.val_graphs
        :property:

        有效的图表ID。

        返回：
            numpy.ndarray，校验图ID数组。

    .. py:method:: mindspore_gl.dataset.Enzymes.val_mask
        :property:

        校验节点掩码。

        返回：
            numpy.ndarray，掩码数组。
