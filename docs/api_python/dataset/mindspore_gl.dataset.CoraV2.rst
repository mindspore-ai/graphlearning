mindspore_gl.dataset.CoraV2
===========================

.. py:class:: mindspore_gl.dataset.CoraV2(root, name='cora_v2')

    Cora Dataset，用于读取和解析Cora数据集的源数据集。

    有关Cora数据集：

    Cora数据集包括2708份科学出版物，分为七类。引文网络由10556个链接组成。数据集中的每个发布都由0/1-valued单词向量描述，指示词典中相应单词的不存在/存在。该词典由1433个独特的单词组成。

    数据：

    - 节点: 2708
    - 边: 10556
    - 分类数量: 7
    - 标签分类:

      - Train: 140
      - Valid: 500
      - Test: 1000

    下载地址：

    `cora_v2 <https://data.dgl.ai/dataset/cora_v2.zip>`_

    `citeseer <https://data.dgl.ai/dataset/citeseer.zip>`_

    `pubmed <https://data.dgl.ai/dataset/pubmed.zip>`_

    您可以将数据集文件组织到以下目录结构中进行读取。

    .. code-block::

        .
        └── corav2
            ├── ind.cora_v2.allx
            ├── ind.cora_v2.ally
            ├── ind.cora_v2.graph
            ├── ind.cora_v2.test.index
            ├── ind.cora_v2.tx
            ├── ind.cora_v2.ty
            ├── ind.cora_v2.x
            └── ind.cora_v2.y

    参数：
        - **root** (str) - 包含cora_v2_with_mask.npz的根目录的路径。
        - **name** (str, 可选) - 选择数据集类型，可选值为 ``"cora_v2"``、 ``"citeseer"``、 ``"pubmed"``。默认值：``"cora_v2"``。

          - ``cora_v2``: 机器学习论文。

          - ``citeseer``: Agents、AI、DB、IR、ML和HCI领域的论文。

          - ``pubmed``: 关于糖尿病的科学出版物。

    异常：
        - **RuntimeError** - 如果 `root` 不包含数据文件。

    .. py:method:: mindspore_gl.dataset.CoraV2.adj_coo
        :property:

        返回COO表示的邻接矩阵。

        返回：
            numpy.ndarray，COO矩阵数组。

    .. py:method:: mindspore_gl.dataset.CoraV2.adj_csr
        :property:

        返回CSR表示的邻接矩阵。

        返回：
            numpy.ndarray，CSR矩阵的数组。

    .. py:method:: mindspore_gl.dataset.CoraV2.edge_count
        :property:

        边数，CSR列的长度。

        返回：
            int，边的数量。

    .. py:method:: mindspore_gl.dataset.CoraV2.node_count
        :property:

        节点数，CSR行的长度。

        返回：
            int，节点的数量。

    .. py:method:: mindspore_gl.dataset.CoraV2.node_feat
        :property:

        节点特征。

        返回：
            numpy.ndarray，节点特征数组。

    .. py:method:: mindspore_gl.dataset.CoraV2.node_feat_size
        :property:

        每个节点的特征维度。

        返回：
            int，特征的维度。

    .. py:method:: mindspore_gl.dataset.CoraV2.node_label
        :property:

        每个节点的真实标签。

        返回：
            numpy.ndarray，节点标签数组。

    .. py:method:: mindspore_gl.dataset.CoraV2.num_classes
        :property:

        标签类数量。

        返回：
            int，类的数量。

    .. py:method:: mindspore_gl.dataset.CoraV2.test_mask
        :property:

        测试节点掩码。

        返回：
            numpy.ndarray，掩码数组。

    .. py:method:: mindspore_gl.dataset.CoraV2.train_mask
        :property:

        训练节点掩码。

        返回：
            numpy.ndarray，掩码数组。

    .. py:method:: mindspore_gl.dataset.CoraV2.train_nodes
        :property:

        训练节点索引。

        返回：
            numpy.ndarray，训练节点数组。

    .. py:method:: mindspore_gl.dataset.CoraV2.val_mask
        :property:

        校验节点掩码。

        返回：
            numpy.ndarray，掩码数组。
