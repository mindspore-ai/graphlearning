mindspore_gl.dataset.BlogCatalog
================================

.. py:class:: mindspore_gl.dataset.BlogCatalog(root)

    BlogCatalog数据集，用于读取和解析BlogCatalog数据集的源数据集。

    参数：
        - **root** (str) - 包含BlogCatalog.npz的根目录的路径。

    异常：
        - **TypeError** - 如果 `root` 不是str。
        - **RuntimeError** - 如果 `root` 不包含数据文件。

    关于BlogCatalog数据集：

    这是从BlogCatalog(http://www.blogcatlog.com)爬取的数据集。BlogCatalog是一个社交博客目录网站，其中包含已爬取的友谊网络和组成员资格。为了便于理解，所有内容都以CSV文件格式组织。

    信息统计：

    - 节点: 10,312
    - Edges: 333,983
    - 类数量: 39

    下载地址：`BlogCatalog <https://figshare.com/articles/dataset/BlogCatalog_dataset/11923611>`_ 。
    您可以将数据集文件组织到以下目录结构中，并通过 `preprocess` API读取。

    .. code-block::

        .
        └── ppi
            ├── edges.csv
            ├── group-edges.csv
            ├── groups.csv
            └── nodes.csv

    .. py:method:: mindspore_gl.dataset.BlogCatalog.preprocess()

        处理数据。

    .. py:method:: mindspore_gl.dataset.BlogCatalog.load()

        从文件加载保存的npz数据集。

    .. py:method:: mindspore_gl.dataset.BlogCatalog.num_classes
        :property:

        标签种类数量。

        返回：
            int，种类量。

    .. py:method:: mindspore_gl.dataset.BlogCatalog.node_count
        :property:

        节点数。

        返回：
            int，csr行的长度。

    .. py:method:: mindspore_gl.dataset.BlogCatalog.edge_count
        :property:

        边数。

        返回：
            int，csr列的长度。

    .. py:method:: mindspore_gl.dataset.BlogCatalog.node_label
        :property:

        基于每个节点的真实标签。

        返回：
            numpy.ndarray，节点标签数组。

    .. py:method:: mindspore_gl.dataset.BlogCatalog.vocab
        :property:

        各节点ID。

        返回：
            numpy.ndarray，节点ID数组。

    .. py:method:: mindspore_gl.dataset.BlogCatalog.adj_coo
        :property:

        返回COO表示的邻接矩阵。

        返回：
            numpy.ndarray，coo矩阵数组。

    .. py:method:: mindspore_gl.dataset.BlogCatalog.adj_csr
        :property:

        返回CSR表示的邻接矩阵。

        返回：
            numpy.ndarray，csr矩阵的数组。
