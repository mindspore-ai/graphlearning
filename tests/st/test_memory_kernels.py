# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" test memory kernels """
import time
from multiprocessing import Process
import numpy as np
import mindspore_gl.memory_kernel as memory_kernel
from mindspore_gl.dataloader import shared_numpy
import pytest


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_memory_kernel_single_process():
    """ test memory kernel single process """
    shared_memory = shared_numpy.SharedMemory(create=True, size=40)

    ref_count = memory_kernel.py_ref_count(shared_memory.buf)
    assert ref_count == 0

    ##################
    # Init RefCount
    ##################
    memory_kernel.py_init_refcount(shared_memory.buf)
    ref_count = memory_kernel.py_ref_count(shared_memory.buf)
    assert ref_count == 1

    ##################
    # Incr RefCount
    ##################
    memory_kernel.py_inc_ref(shared_memory.buf)
    ref_count = memory_kernel.py_ref_count(shared_memory.buf)
    assert ref_count == 2

    ##################
    # Decr RefCount
    ##################
    memory_kernel.py_dec_ref(shared_memory.buf)
    ref_count = memory_kernel.py_ref_count(shared_memory.buf)
    assert ref_count == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_memory_kernel_through_queue_single_process():
    """ test memory kernel through queue single process """
    array = np.ones([1000, 100], dtype=np.int32)
    shared_array = shared_numpy.SharedNDArray.from_numpy_array(array)
    queue = shared_numpy.Queue()
    #########################
    # Data Through Queue
    ########################
    queue.put(shared_array)
    out: shared_numpy.SharedNDArray = queue.get()

    ###########################
    # Check Reference Count
    ##########################
    ref_count = memory_kernel.py_ref_count(out.shm.buf)
    assert ref_count == 2
    ref_count = memory_kernel.py_ref_count(shared_array.shm.buf)
    assert ref_count == 2
    del out
    ref_count = memory_kernel.py_ref_count(shared_array.shm.buf)
    assert ref_count == 1

    ####################
    # Unlink Shm
    ####################
    del shared_array


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_memory_kernel_through_queue_multi_process():
    """ test memory kernel through queue multi process """
    queue = shared_numpy.Queue()

    def producer(q: shared_numpy.Queue):
        array = np.ones([1000, 100], dtype=np.int32)
        shared_array = shared_numpy.SharedNDArray.from_numpy_array(array)
        ############################################################################
        # This Happens Asynchronously, We Should Be Cautious When Reuse Memory
        # But Fortunately, This Wont Affect Us During Training/Inference
        ############################################################################
        q.put(shared_array)
        time.sleep(0.0000001)
        ref_count = memory_kernel.py_ref_count(shared_array.shm.buf)
        assert ref_count == 2
        time.sleep(0.05)
        ref_count = memory_kernel.py_ref_count(shared_array.shm.buf)
        assert ref_count == 1

    def consumer(q: shared_numpy.Queue):
        out: shared_numpy.SharedNDArray = q.get()
        ref_count = memory_kernel.py_ref_count(out.shm.buf)
        assert ref_count == 2
        time.sleep(0.1)
        del out

    consumer_process = Process(target=consumer, args=(queue,))
    producer_process = Process(target=producer, args=(queue,))

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()
