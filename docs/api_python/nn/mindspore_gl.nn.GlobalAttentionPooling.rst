mindspore_gl.nn.GlobalAttentionPooling
======================================

.. py:class:: mindspore_gl.nn.GlobalAttentionPooling(gate_nn, feat_nn=None)

    将全局注意力池应用于图表中的节点。
    来自论文 `Gated Graph Sequence Neural Networks <https://arxiv.org/pdf/1511.05493.pdf>`_ 。

    .. math::
        r^{(i)} = \sum_{k=1}^{N_i}\mathrm{softmax}\left(f_{gate}
            \left(x^{(i)}_k\right)\right) f_{feat}\left(x^{(i)}_k\right)

    参数：
        - **gate_nn** (Cell) - 用于计算每个特征的注意力分数的神经网络。
        - **feat_nn** (Cell) - 在将每个特征与注意力分数结合起来之前，应用于每个特征的神经网络。默认值：None。

    输入：
        - **x** (Tensor) - 要更新的输入节点特征。Shape为 :math:`(N, D)`
          其中 :math:`N` 是节点数，:math:`D` 是节点的特征大小。
        - **g** (BatchedGraph) - 输入图。

    输出：
        - **x** (Tensor) - 图的输出表示。Shape为 :math:`2, D_{out}`
          其中 :math:`D_{out}` 是节点的特征大小。

    异常：
        - **TypeError** - 如果 `gate_nn` 类型或 `feat_nn` 类型不是mindspore.nn.Cell。
