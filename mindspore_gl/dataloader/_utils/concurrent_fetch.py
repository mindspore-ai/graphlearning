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
"""concurrent fetch loop"""
import queue
MP_STATUS_CHECK_INTERVAL = 5.0
def _concurrent_fetch_loop(in_queue, out_queue, done_event):
    """concurrently fetch data in another thread"""

    while not done_event.is_set():
        try:
            r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            continue
        while not done_event.is_set():
            try:
                out_queue.put(r, timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Full:
                continue
        del r  # save memory
