mindspore_gl.dataset.MetrLa
===========================

.. py:class:: mindspore_gl.dataset.MetrLa(root)

    METR-LA，用于读取和解析METR-LA数据集的源数据集。

    关于METR-LA数据集：

    METR-LA是一个大规模数据集，由1500个交通环路探测器在洛杉矶乡村道路网络中收集而来。该数据集包括速度、流量和占用率数据，涵盖了大约3420英里的道路。

    数据：

    - 时间步: 12,6850
    - 节点: 207
    - 边: 1515

    下载地址：`METR-LA <https://graphmining.ai/temporal_datasets/METR-LA.zip>`_ 。您可以将数据集文件组织到以下目录结构中，并通过 `mindspore_gl.dataset.MetrLa.get_data` API读取。

    .. code-block::

        .
        ├── adj_mat.npy
        └── node_values.npy

    参数：
        - **root** (str) - 包含METR-LA/adj_mat.npy和METR-LA/node_values.npy的根目录路径。

    输入：
        - **in_timestep** (int) - 输入时序数。
        - **out_timestep** (int) - 输出时序数。

    异常：
        - **TypeError** - 如果 `root` 不是str。
        - **RuntimeError** - 如果 `root` 不包含数据文件。
        - **TypeError** - 如果 `in_timestep` 或 `out_timestep` 不是正整数。

    .. py:method:: mindspore_gl.dataset.MetrLa.get_data(in_timestep, out_timestep)

        获取序列时间特征和标签。

        参数：
            - **in_timestep** (int) - 输入时序数。
            - **out_timestep** (int) - 输出时序数。

    .. py:method:: mindspore_gl.dataset.MetrLa.node_count
        :property:

        节点数。

        返回：
            int，节点数。