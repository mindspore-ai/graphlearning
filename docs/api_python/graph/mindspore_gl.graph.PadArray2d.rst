mindspore_gl.graph.PadArray2d
=============================

.. py:class:: mindspore_gl.graph.PadArray2d(dtype, direction, fill_value=None, reset_with_fill_value=True, mode=PadMode.AUTO, size=None, use_shared_numpy=False)

    PadArray2d。二维数组的特定pad算子。

    .. warning::
        PadArray2d将重用内存缓冲区以加快Pad操作。

    参数：
        - **dtype** (numpy.dtype) - 决定结果的数据类型。
        - **direction** (PadDirection) - 阵列的pad方向，PadDirection。ROW表示我们将沿着1轴填充，PadDirection.COl表示将沿着0轴填充。
        - **fill_value** (Union[float, int, None]) - 填充区域的填充值。默认值：None。
        - **reset_with_fill_value** (bool) - PadArray2d将重用内存缓冲区，如果对填充值没有要求，可以将此值设置为False。默认值：True。
        - **mode** (PadMode) - 数组的填充模式，如果PadMode.CANST，则此操作将将数组填充到用户特定的大小。如果PadMode.AUTO，这将根据输入的长度选择填充结果长度。预期长度可以计算为 `2^ceil(log2(input_length))` 。默认值：PadMode.AUTO。
        - **size** (Union[List, Tuple]) - 用户特定大小的填充结果。默认值：None。
        - **use_shared_numpy** (bool) - 如果我们使用SharedNDArray来加快进程间通信。如果在子进程中执行特征收集和特征填充，则建议使用此方法，并且图特征需要进程间通信。默认值：False。

    输入：
        - **input_array** (numpy.array) - 需要填充的输入numpy数组。

    异常：
        - **ValueError** - 当填充模式为PadMode.CONST时，应提供填充值大小。

    .. py:function:: mindspore_gl.graph.PadArray2d.lazy(shape: Union[List, Tuple], **kwargs)

        Lazy数组填充，将只确定填充结果形状，并返回一个具有目标形状的空数组。

        参数：
            - **shape** (Union[List, Tuple]) - 需要填充的输入数组的形状。

        返回：
            memory_buffer(numpy.array)，一个空的numpy数组，具有目标填充形状。
