mindspore_gl.dataset.Alchemy
============================

.. py:class:: mindspore_gl.dataset.Alchemy(root, datasize=10000)

    Alchemy数据集，用于读取和解析Alchemy数据集的源数据集。

    关于Alchemy数据集：

    腾讯量子实验室最近推出了一个新的分子数据集，叫做Alchemy，以促进开发对化学和材料科学有用的新机器学习模型。

    该数据集列出了包含多达12个重原子的具有12个量子力学性质的130000+有机分子（C、N、O、S、F和Cl），取自GDBMedChem数据库。这些属性是使用基于Python的化学模拟框架（PySCF）开源计算化学程序。

    信息统计：

    - 图: 99776
    - 节点: 9.71
    - 边: 10.02
    - 量子力学性质数量: 12

    下载地址：
    - `Alchemy dev <https://alchemy.tencent.com/data/dev_v20190730.zip>`_
    - `Alchemy valid <https://alchemy.tencent.com/data/valid_v20190730.zip>`_

    您可以将数据集文件组织到以下目录结构中，并通过 `preprocess` API读取。

    .. code-block::

        .
        ├── dev
        │ ├── dev_target.csv
        │ └── sdf
        │     ├── atom_10
        │     ├── atom_11
        │     ├── atom_12
        │     └── atom_9
        └── valid
            ├── sdf
            │ ├── atom_11
            │ └── atom_12
            └── valid_target.csv

    参数：
        - **root** (str) - 包含alchemy_with_mask.npz的根目录的路径。
        - **datasize** (int) - 训练数据集大小。默认值：10000。

    异常：
        - **TypeError** - 如果 `root` 不是str。
        - **RuntimeError** - 如果 `root` 不包含数据文件。
        - **ValueError** - 如果 `datasize` 大于99776。

    .. py:method:: mindspore_gl.dataset.Alchemy.edge_feat
        :property:

        边特征。

        返回：
            numpy.ndarray，边特征数组。

    .. py:method:: mindspore_gl.dataset.Alchemy.edge_feat_size
        :property:

        每个边的特征数量。

        返回：
            int，特征的数量。

    .. py:method:: mindspore_gl.dataset.Alchemy.graph_count
        :property:

        图的总数。

        返回：
            int，图的数量。

    .. py:method:: mindspore_gl.dataset.Alchemy.graph_edge_feat(graph_idx)

        图上每个边的特征。

        参数：
            - **graph_idx** (int) - 图索引。

        返回：
            numpy.ndarray，图的边特征。

    .. py:method:: mindspore_gl.dataset.Alchemy.graph_edges
        :property:

        累计图边数。

        返回：
            numpy.ndarray，累积边数组。

    .. py:method:: mindspore_gl.dataset.Alchemy.graph_label
        :property:

        图的标签。

        返回：
            numpy.ndarray，图标签数组。

    .. py:method:: mindspore_gl.dataset.Alchemy.graph_node_feat(graph_idx)

        图上每个节点的特征。

        参数：
            - **graph_idx** (int) - 图索引。

        返回：
            numpy.ndarray，图的节点特征。

    .. py:method:: mindspore_gl.dataset.Alchemy.graph_nodes
        :property:

        累计图节点数。

        返回：
            numpy.ndarray，累计节点数组。

    .. py:method:: mindspore_gl.dataset.Alchemy.node_feat
        :property:

        节点特征。

        返回：
            numpy.ndarray，节点特征数组。

    .. py:method:: mindspore_gl.dataset.Alchemy.node_feat_size
        :property:

        每个节点的特征数量。

        返回：
            int，特征的数量。

    .. py:method:: mindspore_gl.dataset.Alchemy.num_classes
        :property:

        图标签种类。

        返回：
            int，图标签的种类。

    .. py:method:: mindspore_gl.dataset.Alchemy.train_graphs
        :property:

        训练图ID。

        返回：
            numpy.ndarray，训练图ID。

    .. py:method:: mindspore_gl.dataset.Alchemy.train_mask
        :property:

        训练节点掩码。

        返回：
            numpy.ndarray，掩码数组。

    .. py:method:: mindspore_gl.dataset.Alchemy.val_graphs
        :property:

        校验的图ID。

        返回：
            numpy.ndarray，校验图ID数组。

    .. py:method:: mindspore_gl.dataset.Alchemy.val_mask
        :property:

        校验节点掩码。

        返回：
            numpy.ndarray，掩码数组。
