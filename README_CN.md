# MindSpore Graph Learning

- [简介](#简介)
- [安装教程](#安装教程)
    - [确认系统环境信息](#确认系统环境信息)
    - [安装方式](#安装方式)
        - [pip安装](#pip安装)
        - [源码安装](#源码安装)
    - [验证是否成功安装](#验证是否成功安装)
- [社区](#社区)
    - [治理](#治理)
- [贡献](#贡献)
- [许可证](#许可证)

<a href="https://gitee.com/mindspore/docs/blob/master/docs/graphlearning/docs/source_zh_cn/mindspore_graphlearning_install.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>&nbsp;&nbsp;

## 简介

MindSpore Graph Learning 是一个基于MindSpore的高效易用的图学习框架。得益于MindSpore的图算融合能力，MindSpore Graph Learning能够针对图模型特有的执行模式进行编译优化，帮助开发者缩短训练时间。
MindSpore Graph Learning 还创新提出了以点为中心编程范式，提供更原生的图神经网络表达方式，并内置覆盖了大部分应用场景的模型，使开发者能够轻松搭建图神经网络。

![GraphLearning_architecture](./images/MindSpore_GraphLearning_architecture_ch.PNG)

## 安装指南

### 确认系统环境信息

- 硬件平台确认为Linux系统，暂不支持Windows和Mac。
- 参考[MindSpore安装指南](https://www.mindspore.cn/install)，完成MindSpore的安装，要求至少1.6.1版本。
- 其余依赖请参见[requirements.txt](https://gitee.com/mindspore/graphlearning/blob/master/requirements.txt)。

### MindSpore版本依赖关系

由于MindSpore Graph Learning与MindSpore有依赖关系，请按照根据下表中所指示的对应关系，在[MindSpore下载页面](https://www.mindspore.cn/versions)下载并安装对应的whl包。

| MindSpore Graph Learning 版本 |                                分支                                | MindSpore运行最低版本 |
|:---------------------------:|:----------------------------------------------------------------:|:---------------:|
|              master              |    [master](https://gitee.com/mindspore/graphlearning/tree/master/)     |                >=2.0.0                |
|              0.2.0               |   [r0.2.0](https://gitee.com/fengxun705612/graphlearning/tree/r0.2.0)   |                >=2.0.0                |

### 安装方式

可以采用pip安装或者源码编译安装两种方式。

#### pip安装

- Ascend/CPU

    ```bash
    pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.0.0rc1/GraphLearning/cpu/{system_structure}/mindspore_gl-0.2-cp37-cp37m-linux_{system_structure}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

- GPU

    ```bash
    pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.0.0rc1/GraphLearning/gpu/x86_64/cuda-{cuda_verison}/mindspore_gl-0.2-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

> - 在联网状态下，安装whl包时会自动下载MindSpore Graph Learning安装包的依赖项（依赖项详情参见[requirements.txt](https://gitee.com/mindspore/graphlearning/blob/master/requirements.txt)），其余情况需自行安装。
> - `{system_structure}`表示为Linux系统架构，可选项为`x86_64`和`arrch64`。
> - `{cuda_verison}`表示为CUDA版本，可选项为`10.1`、`11.1`和`11.6`。

#### 源码安装

1. 从代码仓下载源码

    ```bash
    git clone https://gitee.com/mindspore/graphlearning.git
    ```

2. 编译安装MindSpore Graph Learning

    ```bash
    cd graphlearning
    bash build.sh
    pip install ./output/mindspore_gl*.whl
    ```

### 验证是否成功安装

执行如下命令，如果没有报错`No module named 'mindspore_gl'`，则说明安装成功。

```bash
python -c 'import mindspore_gl'
```
