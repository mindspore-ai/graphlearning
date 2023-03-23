mindspore_gl.graph.CsrAdj
=========================

.. py:class:: mindspore_gl.graph.CsrAdj(indptr, indices)

    构建CSR矩阵的nametuple。

    参数：
         - **indptr** (numpy.ndarry) - CSR矩阵的indptr。
         - **indices** (numpy.ndarry) - CSR矩阵的indices。

    异常：
        - **TypeError** - 如果 `indptr` 或 `indices` 的类型不是numpy.ndarray。
        - **TypeError** - 如果 `indptr` 或 `indices` 的类型不是一维数组。
