mindspore_gl.nn.GatedGraphConv
==============================

.. py:class:: mindspore_gl.nn.GatedGraphConv(in_feat_size: int, out_feat_size: int, n_steps: int, n_etype: int, bias=True)

    门控图卷积层。来自论文 `Gated Graph Sequence Neural Networks <https://arxiv.org/pdf/1511.05493.pdf>`_ 。

    .. math::
        h_{i}^{0} = [ x_i \| \mathbf{0} ] \\

        a_{i}^{t} = \sum_{j\in\mathcal{N}(i)} W_{e_{ij}} h_{j}^{t} \\

        h_{i}^{t+1} = \mathrm{GRU}(a_{i}^{t}, h_{i}^{t})

    参数：
        - **in_feat_size** (int) - 输入节点特征大小。
        - **out_feat_size** (int) - 输出节点特征大小。
        - **n_steps** (int) - 步骤数。
        - **n_etype** (int) - 边类型的数量。
        - **bias** (bool, 可选) - 是否使用偏置。默认值：True。

    输入：
        - **x** (Tensor) - 输入节点特征。Shape为 :math:`(N,*)`
          其中 :math:`N` 是节点数， :math:`*` 可以是任何shape。
        - **src_idx** (List) - 每个边类型的源索引。
        - **dst_idx** (List) - 每个边类型的目标索引。
        - **n_nodes** (int) - 整个图的节点数。
        - **n_edges** (List) - 每个边类型的边数。

    输出：
        - Tensor，输出节点特征。Shape为 :math:`(N,out\_feat\_size)` 。

    异常：
        - **TypeError** - 如果 `in_feat_size` 不是正整数。
        - **TypeError** - 如果 `out_feat_size` 不是正整数。
        - **TypeError** - 如果 `n_steps` 不是正整数。
        - **TypeError** - 如果 `n_etype` 不是正整数。
        - **TypeError** - 如果 `bias` 不是bool。
