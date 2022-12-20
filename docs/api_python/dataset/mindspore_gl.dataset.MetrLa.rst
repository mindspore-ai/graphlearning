mindspore_gl.dataset.MetrLa
===========================

.. py:class:: mindspore_gl.dataset.MetrLa(root)

    METR-LA，用于读取和解析METR-LA数据集的源数据集。

    参数：
        - **root** (str) - 包含METR-LA/adj_mat.npy和METR-LA/node_values.npy的根目录路径。

    输入：
        - **in_timestep** (int) - 输入时序数。
        - **out_timestep** (int) - 输出时序数。

    异常：
        - **TypeError** - 如果 `root` 不是str。
        - **RuntimeError** - 如果 `root` 不包含数据文件。
        - **TypeError** - 如果 `in_timestep` 或 `out_timestep` 不是正整数。


    有关METR-LA数据集：

    METR-LA是一个大规模数据集，从洛杉矶乡村公路网的1500个交通环路探测器中收集。此数据集包括速度、体积和占用数据，覆盖约3,420英里。

    数据：

    - 时间步: 12,6850
    - 节点: 207
    - 边: 1515

    下载地址：`METR-LA <https://graphmining.ai/temporal_datasets/METR-LA.zip>`_ 。
    您可以将数据集文件组织到以下目录结构中，并通过 `preprocess` API读取。

    .. code-block::

        .
        ├── adj_mat.npy
        └── node_values.npy

    .. py:method:: mindspore_gl.dataset.MetrLa.node_num
        :property:

        节点数。

        返回：
            int，节点数。

    .. py:method:: mindspore_gl.dataset.MetrLa.get_data(in_timestep, out_timestep)

        获取序列时间特征和标签。

        参数：
            - **in_timestep** (int) - 输入时序数。
            - **out_timestep** (int) - 输出时序数。