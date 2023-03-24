mindspore_gl.nn.GNNCell
=======================

.. py:class:: mindspore_gl.nn.GNNCell

    GNN Cell 类。
    默认情况下， `construct` 函数将被翻译成MindSpore可执行的代码。

    .. py:method:: mindspore_gl.nn.GNNCell.disable_display()

        禁用代码对比显示功能。

    .. py:method:: mindspore_gl.nn.GNNCell.enable_display(screen_width=200)

        启用显示代码比较。

        参数：
            - **screen_width** (int, 可选) - 显示代码的屏幕宽度。默认值：200。

    .. py:method:: mindspore_gl.nn.GNNCell.specify_path(path)

        指定构造文件路径。

        参数：
            - **path** (str) - 保存构造文件的路径。

    .. py:method:: sparse_compute(csr=False, backward=False):

        是否采样稀疏算子加速。

        参数：
            - **csr** (bool, 可选) - 是否为CSR图。
            - **backward** (bool, 可选) - 是否采用自定义反向。

        异常：
            - **ValueError** - 如果不是csr算子但是采样自定义反向。
