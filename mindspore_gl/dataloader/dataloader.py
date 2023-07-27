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
"""DataLoader."""
# pylint:disable=R1705,W0212,C0209,W0703,C1801,C0330,C0326
import os
import itertools
import warnings
import tempfile
import errno
import queue
from typing import Any, Callable, TypeVar, Generic, List, Optional
import multiprocessing
import threading
import mindspore.dataset as ds
from . import _utils
from .dataset import Dataset
from .utils import ExceptionWrapper, default_collate
from .shared_numpy import Queue as MultiProcessQueue

Tco = TypeVar('Tco', covariant=True)
T = TypeVar('T')
CollateFn = Callable[[List[T]], Any]


class _DatasetKind:
    @staticmethod
    def create_fetcher(dataset, collate_fn):
        return _utils._MapDatasetFetcher(dataset, collate_fn)  # pylint: disable=W0212


class DataLoader(Generic[Tco]):
    """
    Graph data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.

    Args:
        dataset (Dataset): dataset from which to load the graph data.
        sampler (Sampler or Iterable, optional): defines the strategy to draw
            samples from the dataset.
        num_workers (int, optional): how many subprocesses to use for data loading. ``0`` means that the data will be
            loaded in the main process.
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: ``0``)
        prefetch_factor (int, optional, keyword-only arg): Number of samples loaded
            in advance by each worker. ``2`` means there will be a total of
            2 * num_workers samples prefetched across all workers. (default: ``2``)
        persistent_workers (bool, optional): If ``True``, the data loader will not shutdown
            the worker processes after a dataset has been consumed once. This allows to
            maintain the workers `Dataset` instances alive. (default: ``False``)

    Examples:
        >>> from mindspore_gl.dataloader.dataset import Dataset
        >>> from mindspore_gl.dataloader.samplers import RandomBatchSampler
        >>> from mindspore_gl.dataloader.dataloader import DataLoader
        >>> class Mydataset(Dataset):
            ...def __init__(self, data):
            ......self.data = data
            ......self.len = len(data)
            ............
            ...def __getitem__(self, idx):
            ......if isinstance(idx, list):
            .........value = []
            .........for id in idx:
            ............value.append(self.data[id])
            .........return value
            ......return self.data[idx]
        >>> batch_size =3
        >>> data = list(range(0,12))
        >>> sampler = RandomBatchSampler(data, 3)
        >>> ds = Mydataset(data)
        >>> test_dataloader = DataLoader(ds, sampler)
        >>> print(len(test_dataloader))
            4
    """

    dataset: Dataset[Tco]
    batch_size: Optional[int]
    sampler: ds.Sampler
    num_workers: int
    pin_memory: bool
    drop_last: bool
    collate_fn: Optional[CollateFn]
    timeout: float
    prefetch_factor: int
    persistent_workers: bool
    iterator: Optional['_BaseDataLoaderIter']
    initialized = False

    def __init__(self, dataset: Dataset[Tco], sampler: ds.Sampler,
                 num_workers: int = 0, collate_fn: Optional[CollateFn] = None,
                 timeout: float = 0.0, prefetch_factor: int = 2,
                 persistent_workers: bool = True):

        if not isinstance(num_workers, int) or num_workers < 0:
            raise TypeError("num_workers option should be non-negative; "
                            "use num_workers=0 to disable multiprocessing, "
                            "but got num_workers={}.".format(num_workers))

        if not isinstance(timeout, float) or timeout < 0.0:
            raise TypeError("timeout option should be non-negative float,"
                            "but got timeout={}.".format(timeout))

        if not isinstance(prefetch_factor, int) or prefetch_factor <= 0:
            raise TypeError("prefetch_factor option should be positive, "
                            "but got prefetch_factor={}.".format(prefetch_factor))

        if num_workers == 0 and prefetch_factor != 2:
            raise ValueError('prefetch_factor option could only be specified in multiprocessing.'
                             'let num_workers > 0 to enable multiprocessing.')

        self.dataset = dataset
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.timeout = timeout
        self.iterator = None
        self.sampler = sampler
        if collate_fn is None:
            collate_fn = default_collate
        self.collate_fn = collate_fn
        self.persistent_workers = persistent_workers

        if not isinstance(dataset, Dataset):
            raise TypeError("For dataset, Dataloader expect a Dataset instance, but got {}.".format(dataset))

        if not isinstance(sampler, ds.Sampler):
            raise TypeError("For sampler, DataLoader expect a Sampler instance, but got {}.".format(sampler))

        if not isinstance(collate_fn, Callable):
            raise TypeError("For collate_fn, DataLoader expect a callable function, but got {}.".format(collate_fn))

        if not isinstance(persistent_workers, bool):
            raise TypeError("For persistent_workers, DataLoader expect a bool, but got {}.".format(persistent_workers))

        self.check_worker_number_rationality()

    def _getiterator(self) -> '_BaseDataLoaderIter':
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            self.check_worker_number_rationality()
            return _MultiProcessingDataLoaderIter(self)

    # pylint: disable=C0209
    def __setattr__(self, attr, val):
        if self.initialized and attr in ('batch_size', 'batch_sampler', 'sampler',
                                         'drop_last', 'dataset', 'persistent_workers'):
            raise ValueError('{} attribute should not be set after {} is '
                             'initialized'.format(attr, self.__class__.__name__))

        super().__setattr__(attr, val)

    def __iter__(self) -> '_BaseDataLoaderIter':

        if self.persistent_workers and self.num_workers > 0:
            if self.iterator is None:
                self.iterator = self._getiterator()
            else:
                self.iterator._reset(self)
            return self.iterator
        else:
            return self._getiterator()

    @property
    def index_sampler(self):
        return self.sampler

    def __len__(self) -> int:
        return len(self.index_sampler)

    def check_worker_number_rationality(self):
        """check whether num_workers are set properly"""

        def _create_warning_msg(num_worker_suggest, num_worker_created, cpuset_checked):

            suggested_max_worker_msg = ((
                "Our suggested max number of worker in current system is {}{}, which is smaller "
                "than what this DataLoader is going to create.").format(
                num_worker_suggest,
                ("" if cpuset_checked else " (`cpuset` is not taken into account)"))
            ) if num_worker_suggest is not None else (
                "DataLoader is not able to compute a suggested max number of worker in current system.")

            warn_msg = (
                "This DataLoader will create {} worker processes in total. {} "
                "Please be aware that excessive worker creation might get DataLoader running slow or even freeze, "
                "lower the worker number to avoid potential slowness/freeze if necessary.").format(
                num_worker_created,
                suggested_max_worker_msg)
            return warn_msg

        if not self.num_workers or self.num_workers == 0:
            return

        # try to compute a suggested max number of worker based on system's resource
        max_num_worker_suggest = None
        cpuset_checked = False
        if hasattr(os, 'sched_getaffinity'):
            try:
                max_num_worker_suggest = len(os.sched_getaffinity(0))
                cpuset_checked = True
            except Exception:
                pass
        if max_num_worker_suggest is None:
            # os.cpu_count() could return Optional[int]
            # get cpu count first and check None in order to satisfy mypy check
            cpu_count = os.cpu_count()
            if cpu_count is not None:
                max_num_worker_suggest = cpu_count

        if max_num_worker_suggest is None:
            warnings.warn(_create_warning_msg(
                max_num_worker_suggest,
                self.num_workers,
                cpuset_checked))
            return

        if self.num_workers > max_num_worker_suggest:
            warnings.warn(_create_warning_msg(
                max_num_worker_suggest,
                self.num_workers,
                cpuset_checked))


class _BaseDataLoaderIter:
    """Base class for DataLoaderIter"""

    def __init__(self, loader: DataLoader):
        self._dataset = loader.dataset
        self._index_sampler = loader.index_sampler
        self._num_workers = loader.num_workers
        self._prefetch_factor = loader.prefetch_factor
        self._timeout = loader.timeout
        self._collate_fn = loader.collate_fn
        self._sampler_iter = iter(self._index_sampler)
        self._persistent_workers = loader.persistent_workers
        self._shutdown = False
        self._num_yielded = 0
        self._profile_name = "enumerate(DataLoader)#{}.__next__".format(self.__class__.__name__)

    def __iter__(self) -> '_BaseDataLoaderIter':
        return self

    # pylint: disable=W0613
    def _reset(self, loader, first_iter=False):
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0

    def _next_index(self):
        return next(self._sampler_iter)  # may raise StopIteration

    def _next_data(self):
        raise NotImplementedError

    def __next__(self) -> Any:
        data = self._next_data()
        self._num_yielded += 1

        return data

    next = __next__  # Python 2 compatibility

    def __len__(self) -> int:
        return len(self._index_sampler)

    def __getstate__(self):
        raise NotImplementedError(f"{self.__class__.__name__} cannot be pickled")


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    """Single Process Dataloader Iter"""

    def __init__(self, loader):
        super().__init__(loader)
        self._dataset_fetcher = _DatasetKind.create_fetcher(self._dataset, self._collate_fn)

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        return data

    def __getstate__(self):
        return None


class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    r"""Iterates once over the DataLoader's dataset, as specified by the sampler"""

    def __init__(self, loader):
        super().__init__(loader)

        if self._num_workers <= 0 or self._prefetch_factor <= 0:
            raise ValueError(f'{_num_workers} or {_prefetch_factor} must be positive integer.')

        self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))
        self._worker_result_queue = MultiProcessQueue()  # type: ignore[var-annotated]
        self._worker_pids_set = False
        self._shutdown = False
        self._workers_done_event = multiprocessing.Event()

        self._index_queues = []
        self._workers = []
        for i in range(self._num_workers):
            index_queue = MultiProcessQueue()
            index_queue.cancel_join_thread()
            w = multiprocessing.Process(
                target=_utils.worker._worker_loop,
                args=(self._dataset, index_queue,
                      self._worker_result_queue, self._workers_done_event,
                      self._collate_fn, i)
            )
            w.daemon = True
            w.start()
            self._index_queues.append(index_queue)
            self._workers.append(w)

        self._fetch_thread_done_event = threading.Event()

        # Queue is not type-annotated
        self._data_queue = queue.Queue()  # type: ignore[var-annotated]
        fetch_thread = threading.Thread(
            target=_utils.concurrent_fetch._concurrent_fetch_loop,
            args=(self._worker_result_queue, self._data_queue,
                  self._fetch_thread_done_event))
        fetch_thread.daemon = True
        fetch_thread.start()

        self._concurrent_fetch_thread = fetch_thread

        self._data_queue = self._data_queue

        # .pid can be None only before process is spawned (not the case, so ignore)
        self._worker_pids_set = True
        self._reset(loader, first_iter=True)

    def _reset(self, loader, first_iter=False):
        super()._reset(loader, first_iter)
        self._send_idx = 0
        self._rcvd_idx = 0

        self._task_info = {}
        self._tasks_outstanding = 0

        self._workers_status = [True for _ in range(self._num_workers)]

        for _ in range(self._prefetch_factor * self._num_workers):
            self._try_put_index()

    def _try_put_index(self):
        """try to assign tasks to workers"""

        try:
            index = self._next_index()
        except StopIteration:
            return
        for _ in range(self._num_workers):  # find the next active worker, if any
            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if self._workers_status[worker_queue_idx]:
                break
        else:
            # not found (i.e., didn't break)
            return

        self._index_queues[worker_queue_idx].put((self._send_idx, index))
        self._task_info[self._send_idx] = (worker_queue_idx,)
        self._tasks_outstanding += 1
        self._send_idx += 1

    def _next_data(self):
        while True:

            while self._rcvd_idx < self._send_idx:
                info = self._task_info[self._rcvd_idx]
                worker_id = info[0]
                if len(info) == 2 or self._workers_status[worker_id]:  # has data or is still active
                    break
                del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                # no valid `self._rcvd_idx` is found (i.e., didn't break)
                if not self._persistent_workers:
                    self._shutdown_workers()
                raise StopIteration

            # Now `self._rcvd_idx` is the batch index we want to fetch

            # Check if the next sample has already been generated
            if len(self._task_info[self._rcvd_idx]) == 2:
                data = self._task_info.pop(self._rcvd_idx)[1]
                return self._process_data(data)

            idx, data = self._get_data()
            self._tasks_outstanding -= 1

            if idx != self._rcvd_idx:
                # store out-of-order samples
                self._task_info[idx] += (data,)
            else:
                del self._task_info[idx]
                return self._process_data(data)

    def _get_data(self):
        """get data from result queue"""
        if self._timeout > 0:
            success, data = self._try_get_data(self._timeout)
            if success:
                return data
            else:
                raise RuntimeError('DataLoader timed out after {} seconds'.format(self._timeout))
        else:
            while True:
                success, data = self._try_get_data()
                if success:
                    return data

    def _try_get_data(self, timeout=_utils.MP_STATUS_CHECK_INTERVAL):
        """try to get data from result queue"""
        try:
            data = self._data_queue.get(timeout=timeout)
            return True, data

        except Exception as e:
            failed_workers = []
            for worker_id, w in enumerate(self._workers):
                if self._workers_status[worker_id] and not w.is_alive():
                    failed_workers.append(w)
                    self._mark_worker_as_unavailable(worker_id)

            if len(failed_workers) > 0:
                pids_str = ', '.join(str(w.pid) for w in failed_workers)
                raise RuntimeError('DataLoader worker (pid(s) {}) exited unexpectedly'.format(pids_str)) from e
            if isinstance(e, queue.Empty):
                return False, None

            try:
                # Raise an exception if we are this close to the FDs limit.
                # Apparently, trying to open only one file is not a sufficient
                # test.
                # See NOTE [ DataLoader on Linux and open files limit ]
                fds_limit_margin = 10
                # pylint: disable=R1732
                _ = [tempfile.NamedTemporaryFile() for i in range(fds_limit_margin)]
            except OSError as e:
                if e.errno == errno.EMFILE:
                    raise RuntimeError(
                        "Too many open files. Communication with the"
                        " workers is no longer possible. Please increase the"
                        " limit using `ulimit -n` in the shell or change the"
                        " sharing strategy by calling"
                        " `torch.multiprocessing.set_sharing_strategy('file_system')`"
                        " at the beginning of your code") from None
            raise

    def _process_data(self, data):
        self._rcvd_idx += 1
        self._try_put_index()
        if isinstance(data, ExceptionWrapper):
            data.reraise()
        return data

    def _mark_worker_as_unavailable(self, worker_id, shutdown=False):
        """mark workers as unavailable so we wont assign tasks to these workers anymore"""

        if not (self._workers_status[worker_id] or (self._persistent_workers and shutdown)):
            raise RuntimeError('DataLoader run is unavailable')

        # Signal termination to that specific worker.
        q = self._index_queues[worker_id]
        # Indicate that no more data will be put on this queue by the current process.
        q.put(None)

        self._workers_status[worker_id] = False

        if self._workers_done_event.is_set() != shutdown:
            raise RuntimeError('DataLoader run is shutdown')

    def _shutdown_workers(self):
        """shutdown all workers by sending signals to them"""

        if not self._shutdown:
            self._shutdown = True
            try:

                self._fetch_thread_done_event.set()
                self._worker_result_queue.put((None, None))
                self._concurrent_fetch_thread.join()
                self._worker_result_queue.cancel_join_thread()
                self._worker_result_queue.close()

                # Exit workers now.
                self._workers_done_event.set()
                for worker_id in range(len(self._workers)):

                    if self._persistent_workers or self._workers_status[worker_id]:
                        self._mark_worker_as_unavailable(worker_id, shutdown=True)

                for w in self._workers:
                    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)

                for q in self._index_queues:
                    q.cancel_join_thread()
                    q.close()
            finally:

                if self._worker_pids_set:
                    self._worker_pids_set = False
                for w in self._workers:
                    if w.is_alive():
                        w.terminate()

    def __del__(self):
        self._shutdown_workers()

    def __getstate__(self):
        return None
