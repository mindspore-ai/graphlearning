mindspore_gl.nn.Set2Set
=======================

.. py:class:: mindspore_gl.nn.Set2Set(input_size, num_iters, num_layers)

    集合中的sequence to sequence。

    来自论文 `Order Matters: Sequence to sequence for sets <https://arxiv.org/abs/1511.06391>`_ 。

    对于批处理图中的每个子图，计算：

    .. math::
        q_t = \mathrm{LSTM} (q^*_{t-1}) \\

        \alpha_{i,t} = \mathrm{softmax}(x_i \cdot q_t) \\

        r_t = \sum_{i=1}^N \alpha_{i,t} x_i\\

        q^*_t = q_t \Vert r_t

    参数：
        - **input_size** (int) - 输入节点特征的维度。
        - **num_iters** (int) - 迭代次数。
        - **num_layers** (int) - 池化层数。

    输入：
        - **x** (Tensor) - 要更新的输入节点特征。Shape为 :math:`(N, D)`，
          其中 :math:`N` 是节点数，:math:`D` 是节点的特征大小。
        - **g** (BatchedGraph) - 输入图。

    输出：
        - **x** (Tensor) - 图形的输出表示。Shape为 :math:`(2, D_{out})`
          其中 :math:`D_{out}` 是节点特征的双倍大小

    异常：
        - **TypeError** - 如果 `input_size` 或 `num_iters` 或 `num_layers` 不是int。
