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
"""Implement various data sampler."""
import random
import mindspore.dataset as ds


class RandomBatchSampler(ds.Sampler):
    """
    Random Batched Node Sampler, random sample nodes form graph. The remained sample will be dropped.

    Args:
        data_source(Union[List, Tuple, Iterable]): data source sample from
        batch_size(int): number of sampling subgraphs per batch

    Raises:
        TypeError: If `batch_size` is not a positive integer.

    Examples:
        >>> from mindspore_gl.dataloader.samplers import RandomBatchSampler
        >>> ds = list(range(10))
        >>> sampler = RandomBatchSampler(ds, 3)
        >>> print(list(sampler))
        # results will be random for suffle
        [[5, 9, 3], [4, 6, 7], [2, 8, 1]]

    """
    def __init__(self, data_source, batch_size):
        super().__init__()
        self.data_source = data_source
        self.batch_size = batch_size
        if self.data_source is None:
            self.data_source = []
        if isinstance(self.data_source, tuple):
            self.data_source = list(self.data_source)

        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise TypeError("batch_size should be a positive integer value,"
                            "but got batch_size = {}.".format(self.batch_size))
        self.epoch = 1

    def _node_iter(self):
        data_length = len(self.data_source)
        for i in range(0, data_length, self.batch_size):
            # Drop reminder
            if i + self.batch_size <= data_length:
                yield self.data_source[i: i + self.batch_size]

    def __iter__(self):
        # Reset random seed here if necessary
        self.epoch += 1
        random.seed(self.epoch)
        random.shuffle(self.data_source)
        return self._node_iter()

    def __len__(self):
        return len(self.data_source) // self.batch_size


class DistributeRandomBatchSampler(ds.Sampler):
    """
    Distribute Random Batch Sampler

    Args:
        rank(int): Rank of the current process within distributed group, less than `world_size`
        world_size(int): Number of processes in distributed computing
        data_source(Union[List, Tuple, Iterable]): data source sample from
        batch_size(int): number of sampling subgraphs per batch

    Raises:
        TypeError: If `batch_size` is not a positive integer.
        TypeError: If `rank` is negative or not an integer or `rank` value greater than `work_size`.
        TypeError: If `work_size` is not a positive integer.


    Examples:
        >>> from mindspore_gl.dataloader.samplers import DistributeRandomBatchSampler
        >>> ds = list(range(20))
        >>> rank_id = 0
        >>> world_size = 2
        >>> sampler = DistributeRandomBatchSampler(rank_id, world_size, ds, 3)
        >>> print(list(sampler))
        # results will be random for suffle
        [[10, 18, 6], [8, 12, 14], [4, 16, 2]]

    """
    def __init__(self, rank, world_size, data_source, batch_size):
        super().__init__()
        if data_source is None:
            data_source = []
        if isinstance(data_source, tuple):
            data_source = list(data_source)

        self.data_source_rank = data_source[rank::world_size]
        self.batch_size = batch_size
        self.epoch = 1
        self.rank = rank
        self.world_size = world_size
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise TypeError("batch_size should be a positive integer value,"
                            "but got batch_size = {}.".format(self.batch_size))
        if not isinstance(self.world_size, int) or self.world_size < 0:
            raise TypeError("world_size should be a positive integer value,"
                            "but got world_size = {}.".format(self.world_size))
        if not isinstance(self.rank, int) or self.rank < 0 or self.rank >= self.world_size:
            raise TypeError("rank should be a positive integer value less than work_size,"
                            "but got rank = {}.".format(self.rank))

    def node_iter(self):
        data_length = len(self.data_source_rank)
        for i in range(0, data_length, self.batch_size):
            # Drop reminder
            if i + self.batch_size <= data_length:
                yield self.data_source_rank[i: i + self.batch_size]

    def __iter__(self):
        # Reset random seed here if necessary
        self.epoch += 1
        random.seed(self.epoch)
        random.shuffle(self.data_source_rank)
        return self.node_iter()

    def __len__(self):
        return (len(self.data_source_rank) + self.batch_size - 1) // self.batch_size
