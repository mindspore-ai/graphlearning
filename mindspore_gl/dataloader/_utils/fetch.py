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
"""Fetcher"""


class _BaseDatasetFetcher:
    """BaseDatasetFetcher"""
    def __init__(self, dataset, collate_fn):
        self.dataset = dataset
        self.collate_fn = collate_fn

    def fetch(self, possibly_batched_index):
        raise NotImplementedError()


class _IterableDatasetFetcher(_BaseDatasetFetcher):
    """IterableDatasetFetcher"""

    def __init__(self, dataset, collate_fn):
        super().__init__(dataset, collate_fn)
        self.dataset_iter = iter(dataset)
        self.ended = False

    def fetch(self, possibly_batched_index):
        if self.ended:
            raise StopIteration
        data = next(self.dataset_iter)
        return self.collate_fn(data)


class _MapDatasetFetcher(_BaseDatasetFetcher):
    """MapDatasetFetcher"""
    def __init__(self, dataset, collate_fn):#pylint:disable=W0235
        super().__init__(dataset, collate_fn) #pylint:disable=W0235

    def fetch(self, possibly_batched_index):
        data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)
