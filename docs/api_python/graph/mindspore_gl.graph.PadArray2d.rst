mindspore_gl.graph.PadArray2d
=============================

.. py:class:: mindspore_gl.graph.PadArray2d(dtype, direction, fill_value=None, reset_with_fill_value=True, mode=PadMode.AUTO, size=None, use_shared_numpy=False)

    PadArray2d。二维数组的特定pad算子。

    .. warning::
        PadArray2d将重用内存缓冲区以加快Pad操作。

    参数：
        - **dtype** (numpy.dtype) - 决定结果的数据类型。
        - **direction** (PadDirection) - 阵列的pad方向，``PadDirection.ROW`` 表示我们将沿着1轴填充，``PadDirection.COL`` 表示将沿着0轴填充。
        - **fill_value** (Union[float, int, 可选]) - 填充区域的填充值。默认值：``None``。
        - **reset_with_fill_value** (bool, 可选) - PadArray2d将重用内存缓冲区，如果对填充值没有要求，可以将此值设置为 ``False``。默认值：``True``。
        - **mode** (PadMode, 可选) - 数组的填充模式，如果 ``PadMode.CANST``，则此操作将将数组填充到用户特定的大小。如果 ``PadMode.AUTO``，这将根据输入的长度选择填充结果长度。预期长度可以计算为 :math:`length=2^{\text{ceil}\left ( \log_{2}{input\_length}  \right ) }`。默认值： ``mindspore_gl.graph.PadMode.AUTO``。
        - **size** (Union[List, Tuple, 可选]) - 用户特定大小的填充结果。默认值：``None``。
        - **use_shared_numpy** (bool, 可选) - 是否使用SharedNDArray来加快进程间通信。如果在子进程中执行特征收集和特征填充，则建议使用此方法，并且图特征需要进程间通信。默认值：``False``。

    输入：
        - **input_array** (numpy.array) - 需要填充的输入numpy数组。

    异常：
        - **ValueError** - 当填充模式为 ``PadMode.CONST`` 时，应提供填充值大小。

    .. py:method:: mindspore_gl.graph.PadArray2d.lazy(shape: Union[List, Tuple], **kwargs)

        Lazy数组填充，将只确定填充结果shape，并返回一个具有目标shape的空数组。

        参数：
            - **shape** (Union[List, Tuple]) - 需要填充的输入数组的shape。
            - **kwargs** (dict) - 配置选项字典。
              - **fill_value** (Union[int, float]) - 配置选项字典关键字参数。

        返回：
            memory_buffer(numpy.array)，一个空的numpy数组，具有目标填充shape。
