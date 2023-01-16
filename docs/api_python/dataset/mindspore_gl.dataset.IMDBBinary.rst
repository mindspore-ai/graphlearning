mindspore_gl.dataset.IMDBBinary
===============================

.. py:class:: mindspore_gl.dataset.IMDBBinary(root)

    IMDBBinary数据集，用于读取和解析IMDBBinary数据集的源数据集。

    关于IMDBBinary数据集：

    IMDBBinary数据集，用于读取和解析IMDBBinaary数据集的源数据集。IMDB-BINARY是一个电影协作数据集，由1000名在IMDB电影中扮演角色的演员组成的角色扮演网络组成。
    在每个图中，节点表示演员/女演员，如果他们出现在同一部电影中，则它们之间有一条边。这些图来源于动作片和浪漫片。

    信息统计：

    - 节点: 19773
    - 边: 193062
    - 图： 1000
    - 分类数: 2
    - 数据集切分:

      - Train: 800
      - Valid: 200

    下载地址：`<https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/IMDB-BINARY.zip>`_ 。
    您可以将数据集文件组织到以下目录结构中，并通过 `preprocess` API读取。

    .. code-block::

        .
        ├── IMDB-BINARY_A.txt
        ├── IMDB-BINARY_graph_indicator.txt
        └── IMDB-BINARY_graph_labels.txt

    参数：
        - **root** (str) - 包含imdb_binary_with_mask.npz的根目录的路径。

    异常：
        - **TypeError** - 如果 `root` 不是str。
        - **RuntimeError** - 如果 `root` 不包含数据文件。

    .. py:method:: mindspore_gl.dataset.IMDBBinary.graph_count
        :property:

        图的总数。

        返回：
            int，图的数量。

    .. py:method:: mindspore_gl.dataset.IMDBBinary.graph_edges
        :property:

        累计图边数。

        返回：
            numpy.ndarray，累积边数组。

    .. py:method:: mindspore_gl.dataset.IMDBBinary.graph_label
        :property:

        图的标签。

        返回：
            numpy.ndarray，图标签数组。

    .. py:method:: mindspore_gl.dataset.IMDBBinary.graph_nodes
        :property:

        累计图节点数。

        返回：
            numpy.ndarray，累计节点数组。

    .. py:method:: mindspore_gl.dataset.IMDBBinary.node_feat
        :property:

        节点特征。

        返回：
            numpy.ndarray，节点特征数组。

    .. py:method:: mindspore_gl.dataset.IMDBBinary.num_classes
        :property:

        图标签种类。

        返回：
            int，图标签的种类。

    .. py:method:: mindspore_gl.dataset.IMDBBinary.edge_feat_size
        :property:

        标签类数量。

        返回：
            int，类的数量。

    .. py:method:: mindspore_gl.dataset.IMDBBinary.node_feat_size
        :property:

        每个节点的特征数量。

        返回：
            int，特征的数量。

    .. py:method:: mindspore_gl.dataset.IMDBBinary.train_graphs
        :property:

        训练图ID。

        返回：
            numpy.ndarray，训练图ID。

    .. py:method:: mindspore_gl.dataset.IMDBBinary.train_mask
        :property:

        训练节点掩码。

        返回：
            numpy.ndarray，掩码数组。

    .. py:method:: mindspore_gl.dataset.IMDBBinary.val_graphs
        :property:

        校验的图ID。

        返回：
            numpy.ndarray，校验图ID数组。

    .. py:method:: mindspore_gl.dataset.IMDBBinary.val_mask
        :property:

        校验节点掩码。

        返回：
            numpy.ndarray，掩码数组。
