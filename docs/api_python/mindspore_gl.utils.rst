mindspore_gl.utils
==================

utils初始化

.. py:function:: mindspore_gl.utils.pca(matrix: np.ndarray, k: int = None, niter: int = 2, norm: bool = False)

    对矩阵执行线性主成分分析（PCA），并将返回前k个降维特征。

    参数：
        - **matrix** (ndarray) - 输入特征，shape为 :math:`(B, F)` 。
        - **k** (int, 可选) - 降维的目标维度。默认值：None。
        - **niter** (int, 可选) - 要进行的子空间迭代次数并且必须是非负整数。默认值：2。
        - **norm** (bool, 可选) - 输出是否归一化。默认值：False。

    返回：
        ndarray，降维后的特征。

    异常：
        - **TypeError** - 如果 `k` 或 `niter` 不是正整数。
        - **TypeError** - 如果 `matrix` 不是numpy.ndarry。
        - **TypeError** - 如果 `norm` 不是bool。

.. automodule:: mindspore_gl.utils
    :members: