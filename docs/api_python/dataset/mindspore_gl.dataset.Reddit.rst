mindspore_gl.dataset.Reddit
===========================

.. py:class:: mindspore_gl.dataset.Reddit(root)

    Reddit 数据集，用于读取和解析Reddit数据集的源数据集。

    有关Reddit数据集：

    节点标签是社区，或帖子所属的“subreddit”。
    作者对50个大型社区进行了抽样调查，并建立了一个post-to-post的图表，连接那些同一用户对两者发表评论的帖子。此数据集总共包含232,965个。
    平均degree为492。我们利用头20天进行训练，剩余天数用于测试（30%用于验证）。

    数据：

    - 节点: 232,965
    - 边: 114,615,892
    - 分类数量: 41

    下载地址：`Reddit <https://data.dgl.ai/dataset/reddit.zip>`_ 。
    您可以将数据集文件组织到以下目录结构中进行读取。

    .. code-block::

        .
        ├── reddit_data.npz
        └── reddit_graph.npz

    参数：
        - **root** (str) - 包含reddit_with_mask.npz的根目录路径。

    异常：
        - **TypeError** - 如果 `root` 不是str。
        - **RuntimeError** - 如果 `root` 不包含数据文件。

    .. py:method:: mindspore_gl.dataset.Reddit.adj_coo
        :property:

        返回COO表示的邻接矩阵。

        返回：
            numpy.ndarray，COO矩阵数组。

    .. py:method:: mindspore_gl.dataset.Reddit.adj_csr
        :property:

        返回CSR表示的邻接矩阵。

        返回：
            numpy.ndarray，CSR矩阵数组。

    .. py:method:: mindspore_gl.dataset.Reddit.edge_count
        :property:

        边的数量，CSR列的长度。

        返回：
            int，边的数量。

    .. py:method:: mindspore_gl.dataset.Reddit.node_count
        :property:

        节点数量，CSR行的长度。

        返回：
            int，节点数量。

    .. py:method:: mindspore_gl.dataset.Reddit.node_feat
        :property:

        节点特征。

        返回：
            numpy.ndarray，节点特征数组。

    .. py:method:: mindspore_gl.dataset.Reddit.node_feat_size
        :property:

        每个节点的特征数量。

        返回：
            int，特征的数量。

    .. py:method:: mindspore_gl.dataset.Reddit.node_label
        :property:

        每个节点的标签。

        返回：
            numpy.ndarray，节点标签的数组。

    .. py:method:: mindspore_gl.dataset.Reddit.num_classes
        :property:

        标签类的数量。

        返回：
            int，分类数。

    .. py:method:: mindspore_gl.dataset.Reddit.test_mask
        :property:

        测试节点掩码。

        返回：
            numpy.ndarray，掩码数组。

    .. py:method:: mindspore_gl.dataset.Reddit.test_nodes
        :property:

        测试节点索引。

        返回：
            numpy.ndarray，测试节点的数组。

    .. py:method:: mindspore_gl.dataset.Reddit.train_mask
        :property:

        训练节点掩码。

        返回：
            numpy.ndarray，掩码数组。

    .. py:method:: mindspore_gl.dataset.Reddit.train_nodes
        :property:

        训练节点索引。

        返回：
            numpy.ndarray，训练节点的数组。

    .. py:method:: mindspore_gl.dataset.Reddit.val_mask
        :property:

        校验节点掩码。

        返回：
            numpy.ndarray，掩码数组。

    .. py:method:: mindspore_gl.dataset.Reddit.val_nodes
        :property:

        验证节点索引。

        返回：
            numpy.ndarray，验证节点的数组。
