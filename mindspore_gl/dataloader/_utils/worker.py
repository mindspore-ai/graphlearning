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

"""worker loop"""
#pylint: disable=C0209
#pylint: disable=W0703
#pylint: disable=R1723

import os
import queue
from .fetch import _MapDatasetFetcher
from ..utils import ExceptionWrapper


class _DatasetKind:
    @staticmethod
    def create_fetcher(dataset, collate_fn):
        return _MapDatasetFetcher(dataset, collate_fn) #pylint: disable=W0212


MP_STATUS_CHECK_INTERVAL = 5.0


class ManagerWatchdog:
    def __init__(self):
        self.manager_pid = os.getppid()
        self.manager_dead = False

    def is_alive(self):
        if not self.manager_dead:
            self.manager_dead = os.getppid() != self.manager_pid
        return not self.manager_dead


def _worker_loop(dataset, index_queue, data_queue, done_event, collate_fn, worker_id):
    """worker loop"""

    try:

        init_exception = None

        try:
            fetcher = _DatasetKind.create_fetcher(dataset, collate_fn)

        except Exception: #pylint:disable=W0703
            init_exception = ExceptionWrapper(
                where="in DataLoader worker process {}".format(worker_id))

        iteration_end = False
        watchdog = ManagerWatchdog()

        while watchdog.is_alive():
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue
            if r is None:
                # Received the final signal
                assert done_event.is_set() or iteration_end
                break
            elif done_event.is_set() or iteration_end:
                # `done_event` is set. But I haven't received the final signal
                # (None) yet. I will keep continuing until get it, and skip the
                # processing steps.
                continue
            idx, index = r
            data = None
            if init_exception is not None:
                data = init_exception
                init_exception = None
            else:
                try:
                    data = fetcher.fetch(index)
                except Exception as e: #pylint: disable=W0703
                    print(e)
                    data = ExceptionWrapper(e, where="in DataLoader worker process {}".format(worker_id))
            data_queue.put((idx, data))
            del data, idx, index, r  # save memory
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass
    if done_event.is_set():
        data_queue.cancel_join_thread()
        data_queue.close()
