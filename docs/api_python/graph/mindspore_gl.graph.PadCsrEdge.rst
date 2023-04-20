mindspore_gl.graph.PadCsrEdge
================================

.. py:class:: mindspore_gl.graph.PadCsrEdge(pad_nodes, reset_with_fill_value=True, length=None, mode=PadMode.AUTO, use_shared_numpy=False)

    特定的COO边填充算子。填充后COO索引转换到到csr时，indices和indptr的shape变得统一。

    .. warning::
        PadArray2d将会重新使用内存缓存来加速填充操作。

    参数：
        - **pad_nodes** (int) - 需要填充的图的节点数。
        - **reset_with_fill_value** (bool, 可选) - PadArray2d将重用内存缓冲区，如果对填充值没有要求，可以将此值设置为 ``False``。默认值：``True``。
        - **length** (int, 可选) - 用户特定大小的填充长度。默认值：``None``。
        - **mode** (PadMode, 可选) - 数组的填充模式，如果 ``PadMode.CANST``，则此操作将将数组填充到用户特定的大小。如果 ``PadMode.AUTO``，这将根据输入的长度选择填充结果长度。预期长度可以计算为
          :math:`length=2^{\text{ceil}\left ( \log_{2}{input\_length}  \right ) }`
          默认值：``mindspore_gl.graph.PadMode.AUTO``。
        - **use_shared_numpy** (bool, 可选) - 是否使用SharedNDArray来加快进程间通信。如果在子进程中执行特征收集和特征填充，则建议使用此方法，并且图特征需要进程间通信。默认值：``False``。

    输入：
        - **input_array** (numpy.array) - 需要填充的输入numpy数组。

    异常：
        - **ValueError** - 当填充模式为 ``PadMode.CONST`` 时，应提供填充值大小。
