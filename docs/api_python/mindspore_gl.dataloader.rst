mindspore_gl.dataloader
=======================

.. py:class:: mindspore_gl.dataloader.Dataset()

    可映射数据集定义，用抽象类表示数据集。
    所有数据集都应该是它的子类，它代表一个从key到样本的映射关系。
    所有子类都应该重写 `__getitem__`，实现根据key来获取样本。

.. py:class:: mindspore_gl.dataloader.RandomBatchSampler(data_source, batch_size)

    随机批处理节点采样器。随机采样节点形成图形。残留的样本将被丢弃。

    参数：
        - **data_source** (Union[List, Tuple, Iterable]) - 采样数据的来源。
        - **batch_size** (int) - 每批次采样子图的数量。

    异常：
        - **TypeError** - 如果 `batch_size` 不是正整数。

.. py:function:: mindspore_gl.dataloader.split_data(x, val_ratio=0.05, test_ratio=0.1, graph_type='undirected')

    根据用户输入的比例，将训练集切割成训练集、验证集和测试集，并对训练集进行图重构，然后返回。

    参数：
        - **x** (mindspore_gl.dataloader.Dataset) - 图结构数据集。
        - **val_ratio** (float, 可选) - 验证集比例。默认值：0.05。
        - **test_ratio** (float, 可选) - 测试集比例。默认值：0.1。
        - **graph_type** (str, 可选) - 图的类型。'undirected'：无向图，'directed'：有向图。默认值：'undirected'。

    返回：
        - **train** (numpy.ndarray) - 训练集，shape :math:`(train\_len, 2)` 。
        - **val** (numpy.ndarray) - 验证集，shape :math:`(val\_len, 2)` 。
        - **test** (numpy.ndarray) - 测试集，shape :math:`(test\_len, 2)` 。

.. automodule:: mindspore_gl.dataloader
    :members:
