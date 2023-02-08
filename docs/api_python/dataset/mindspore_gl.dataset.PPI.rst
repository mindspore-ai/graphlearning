mindspore_gl.dataset.PPI
========================

.. py:class:: mindspore_gl.dataset.PPI(root)

    PPI数据集，用于读取和解析PPI数据集的源数据集。

    有关PPI数据集：

    蛋白质在各种蛋白质-蛋白质相互作用（PPI）图中的作用——就其细胞功能而言——在各种蛋白质-蛋白质相互作用（PPI）图中，每个图对应于不同的人类组织。使用位置基因集，基序基因集和免疫学特征作为特征，基因本体集作为标签（总共121个），从分子特征数据库收集。平均图包含2373个节点，平均度为28.8。

    数据：

    - 图: 24
    - 节点: ~2245.3
    - Edges: ~61,318.4
    - 类的数量: 121
    - 标签分类:

      - Train examples: 20
      - Valid examples: 2
      - Test examples: 2

    下载地址：`PPI <https://data.dgl.ai/dataset/ppi.zip>`_ 。
    您可以将数据集文件组织到以下目录结构中，并通过 `preprocess` API读取。

    .. code-block::

        .
        └── ppi
            ├── valid_feats.npy
            ├── valid_labels.npy
            ├── valid_graph_id.npy
            ├── valid_graph.json
            ├── train_feats.npy
            ├── train_labels.npy
            ├── train_graph_id.npy
            ├── train_graph.json
            ├── test_feats.npy
            ├── test_labels.npy
            ├── test_graph_id.npy
            └── test_graph.json

    参数：
        - **root** (str) - 包含pi_with_mask.npz的根目录路径。

    异常：
        - **TypeError** - 如果 `root` 不是str。
        - **RuntimeError** - 如果 `root` 不包含数据文件。

    .. py:method:: mindspore_gl.dataset.PPI.graph_count
        :property:

        图的总数。

        返回：
            int，图的数量。

    .. py:method:: mindspore_gl.dataset.PPI.graph_edges
        :property:

        累计图边数。

        返回：
            numpy.ndarray，累积边数组。

    .. py:method:: mindspore_gl.dataset.PPI.graph_node_feat(graph_idx)

        图上每个节点的特征。

        参数：
            - **graph_idx** (int) - 图索引。

        返回：
            numpy.ndarray，图的节点特征。

    .. py:method:: mindspore_gl.dataset.PPI.graph_node_label(graph_idx)

        图上每个节点的真实标签。

        参数：
            - **graph_idx** (int) - 图索引。

        返回：
            numpy.ndarray，图的节点标签。

    .. py:method:: mindspore_gl.dataset.PPI.graph_nodes
        :property:

        累计图节点数。

        返回：
            numpy.ndarray，累计节点数组。

    .. py:method:: mindspore_gl.dataset.PPI.node_feat
        :property:

        节点特性。

        返回：
            numpy.ndarray，节点特征数组。

    .. py:method:: mindspore_gl.dataset.PPI.node_label
        :property:

        每个节点的真实标签。

        返回：
            numpy.ndarray，节点标签数组。

    .. py:method:: mindspore_gl.dataset.PPI.num_classes
        :property:

        标签类数量。

        返回：
            int，类的数量。

    .. py:method:: mindspore_gl.dataset.PPI.node_feat_size
        :property:

        每个节点的特征大小。

        返回：
            int，特征大小的数量。

    .. py:method:: mindspore_gl.dataset.PPI.test_graphs
        :property:

        测试图ID。

        返回：
            numpy.ndarray，测试图ID数组。

    .. py:method:: mindspore_gl.dataset.PPI.test_mask
        :property:

        测试节点掩码。

        返回：
            numpy.ndarray，掩码数组。

    .. py:method:: mindspore_gl.dataset.PPI.train_graphs
        :property:

        训练图ID。

        返回：
            numpy.ndarray，训练ID数组。

    .. py:method:: mindspore_gl.dataset.PPI.train_mask
        :property:

        训练节点掩码。

        返回：
            numpy.ndarray，掩码数组。

    .. py:method:: mindspore_gl.dataset.PPI.val_graphs
        :property:

        校验图ID。

        返回：
            numpy.ndarray，校验图ID数组。

    .. py:method:: mindspore_gl.dataset.PPI.val_mask
        :property:

        校验节点掩码。

        返回：
            numpy.ndarray，掩码数组。
