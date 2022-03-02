/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <atomic>
#include <iostream>

static constexpr int64_t map_alloc_alignment = 64;

struct _RefCounter {
  std::atomic<int> refcount;
};

void inc_ref(char* base_ptr) {
    _RefCounter *ref_counter = static_cast<_RefCounter*>(reinterpret_cast<void*>(base_ptr));
    ++ref_counter->refcount;
    return;
}
void dec_ref(char* base_ptr) {
    _RefCounter *ref_counter = static_cast<_RefCounter*>(reinterpret_cast<void*>(base_ptr));
    --ref_counter->refcount;
    return;
}
int ref_count(char *base_ptr) {
    _RefCounter *ref_counter = static_cast<_RefCounter*>(reinterpret_cast<void*>(base_ptr));
    return ref_counter->refcount;
}

void init_refcount(char* base_ptr) {
    _RefCounter *ref_counter = static_cast<_RefCounter*>(reinterpret_cast<void*>(base_ptr));
    new (&ref_counter->refcount) std::atomic<int>(1);
}

