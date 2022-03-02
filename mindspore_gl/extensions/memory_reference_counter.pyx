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

# distutils: language = c++

import numpy as np
cimport numpy as np
cimport cython

cdef extern from "./shared_memory_reference_counter.h":
    void init_refcount(char* )
    int ref_count(char* )
    void inc_ref(char* )
    void dec_ref(char* )

def py_init_refcount(char[::1] buf):
    init_refcount(<char*> &buf[0])


def py_ref_count(char[::1] buf):
    return ref_count(<char*> &buf[0])

def py_inc_ref(char[::1] buf):
    inc_ref(<char*> &buf[0])

def py_dec_ref(char[::1] buf):
    dec_ref(<char*> &buf[0])







