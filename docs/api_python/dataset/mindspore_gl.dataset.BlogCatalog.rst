mindspore_gl.dataset.BlogCatalog
================================

.. py:class:: mindspore_gl.dataset.BlogCatalog(root)

    BlogCatalog数据集，用于读取和解析BlogCatalog数据集的源数据集。

    关于BlogCatalog数据集：

    这是从BlogCatalog获取的数据集。BlogCatalog是一个社交博客目录网站，其中包含友谊网络和小组成员资格。为了便于理解，所有内容都以CSV文件格式组织。

    信息统计：

    - 节点: 10,312
    - Edges: 333,983
    - 分类数量: 39

    下载地址：`BlogCatalog <https://figshare.com/articles/dataset/BlogCatalog_dataset/11923611>`_ 。
    您可以将数据集文件组织到以下目录结构中进行读取。

    .. code-block::

        .
        └── ppi
            ├── edges.csv
            ├── group-edges.csv
            ├── groups.csv
            └── nodes.csv

    参数：
        - **root** (str) - 包含BlogCatalog.npz的根目录的路径。

    异常：
        - **TypeError** - 如果 `root` 不是str。
        - **RuntimeError** - 如果 `root` 不包含数据文件。

    .. py:method:: mindspore_gl.dataset.BlogCatalog.adj_coo
        :property:

        利用COO表示的邻接矩阵。

        返回：
            numpy.ndarray，COO矩阵数组。

    .. py:method:: mindspore_gl.dataset.BlogCatalog.adj_csr
        :property:

        利用CSR表示的邻接矩阵。

        返回：
            numpy.ndarray，CSR矩阵数组。

    .. py:method:: mindspore_gl.dataset.BlogCatalog.edge_count
        :property:

        图的边数，CSR列的长度。

        返回：
            int，图的边数。

    .. py:method:: mindspore_gl.dataset.BlogCatalog.node_count
        :property:

        节点数，CSR行的长度。

        返回：
            int，图的节点数。

    .. py:method:: mindspore_gl.dataset.BlogCatalog.node_label
        :property:

        每个节点的标签。

        返回：
            numpy.ndarray，节点标签数组。

    .. py:method:: mindspore_gl.dataset.BlogCatalog.num_classes
        :property:

        标签种类。

        返回：
            int，标签种类。

    .. py:method:: mindspore_gl.dataset.BlogCatalog.vocab
        :property:

        各个节点的ID。

        返回：
            numpy.ndarray，节点ID数组。
