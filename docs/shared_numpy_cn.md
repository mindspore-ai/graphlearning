# SharedNumpy 特性说明

## 一、动机

基于采样的GNN训练中, 由于单个Batch数据体积较大，如当`Batch_Size = 1024, 2-hop 采样参数为[25, 10]`的情况下，单个Batch理论采样节点总数为 250K，当每个节点上有`602维`
特征时（`Reddit`数据集中节点特征维度），一个Batch的数据大小约为**600M**，这对进程间的通信带来了较大的压力。

## 二、常规方案性能缺点

当前的方案中对于`Numpy Array`的通信都基于序列化和反序列化的传输。然而当数据达到**600M**时,进程间的通信总共耗时大约为**2.29s**, 这显然是我们不能接受的性能。

```python
import numpy as np
import multiprocessing
import time

queue = multiprocessing.Queue()
array = np.ones([250000, 602], dtype=np.float32)

start = time.time()
queue.put(array)
queue.get()
print(f"time consumed by serialization {time.time() - start}")  # 2.29s

```

## 三、基于共享内存方式通信的SharedNDArray的设计与实现

为了优化进程间的通信效率，我们设计并实现了基于共享内存的`SharedNDArray`, `SharedNDArray`支持Numpy Array的所有操作，并可以`zero-copy`的方式生成`mindspore.Tensor`.
基于`SharedNDArray`的进程间通信，其性能提升到了**0.486s**，提升了大约5x。

```python
import numpy as np
import time
from mindspore_gl.dataloader import shared_numpy

queue2 = shared_numpy.Queue()

start = time.time()
shared_array = shared_numpy.SharedNDArray.from_numpy_array(array)
queue2.put(shared_array)
queue2.get()
print(f"time consumed by shared memory {time.time() - start}")  # 0.486s
```

### 3.1 基于引用计数的内存生命周期管理

为了防止进程间共享内存的泄漏导致程序运行时内存占用无限增长，我们使用引用计数的方式进行`SharedNumpyNDArray`内存生命周期的管理。我们在申请内存时，多申请`8B`用于
记录当前内存被各个进程的引用次数。每当该内存被共享给其他进程，则内存块引用计数+1， 当一个进程中的`SharedNumpyNDArray`对象被虚拟机垃圾回收，则引用计数-1，
当最后一个引用该内存的`SharedNumpyNDArray`对象被Python虚拟机进行垃圾回收时，会自动的`unlink`共享内存以方便操作系统回收。 基于上述的方法，我们可以方便高效的实现内存生命周期管理。

## 四、共享内存池的设计实现

### 4.1 动机

有了`SharedNDArray`之后，我们的进程间通信已然高效了许多，但是400ms的耗时仍然是我们所不能接受的，我们于是想进一步减少这部分消耗。

基于共享内存的通信核心有两个步骤:

- 申请一块共享内存区域
- 将数据拷贝到对应区域

数据拷贝不太好省略，但是我们想通过内存池复用来减少对于共享内存块的重复申请和释放。于是我们实现了`SharedNumpyPool`用于进程间的共享加速。当用户想要共享内存时，从`SharedNumpyPool`
中申请内存，如果过往已经申请过同样大小的内存，则会复用该部分内存以避免再次申请。

### 4.2 难点

想象以下场景:

进程`P1`通过共享内存将大小为`Size`数据`Data1`共享给进程`P2`后，`P2`进程中一直在使用该内存进行操作。此时进程`P1`又需要共享同样大小为`Size`的 数据`Data2`给进程`P3`, 此时`P1`
该如何判断是否复用`Data1`原先申请好的内存呢？如果不加任何判断直接将`Data2`的数据拷贝到`Data1`所在的内存中，很可能会导致 正在使用`Data1`的进程`P1`
在运行时由于数据被修改导致出错。于是我们需要设计一套良好的机制来避免这个问题。

### 4.3 解决方案

幸运的是，我们再一开始就设计了基于引用计数的机制来实现共享内存的生命周期管理。那么当`Data2`需要被共享时，P1只需要检查原先`Data1`所在的内存块的引用次数即可。 如果此时引用计数为1，那么说明只有`P1`
目前再引用该内存，其他进程基于该内存的对象已经被清除，则`P1`可以复用该内存，否则说明其他进程中还有对象再使用这块内存， 则`P1`需要重新申请内存给`Data2`，申请后这部分内存会被放入内存池用于日后的复用。

### 4.3 效果

有了该方案后，我们基于复用内存的共享便高效快捷了很多。整个600M的数据内存共享耗时约为`0.120s`， 这相比较Python自带的方案已经快了将近**20**倍。目前已经 达到了我们的性能需求。

```python
import numpy as np
import time
from mindspore_gl.dataloader import shared_numpy

queue2 = shared_numpy.Queue()
array = np.ones([250000, 602], dtype=np.float32)
shared_array = shared_numpy.SharedNDArray.from_numpy_array(array)

# 模拟内存复用，故只需要赋值
start = time.time()
shared_array[:] = array[:]
queue2.put(shared_array)
queue2.get()
print(f"time consumed by shared array pool {time.time() - start}")  # 0.120

```

## 五、总结

如下是我们的优化路线总结。

|优化措施|600M NumpyArray 共享性能|加速收益|
|:----|:----|:----|
|Python自带方案|2.29s| -|
|基于SharedNDArray|0.486s|4.7 x|
|基于SharedNumpyPool|0.120s|19.24 x|





