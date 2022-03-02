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
""" Test dataloader api. """
import numpy as np
import pytest
from mindspore_gl.dataloader.dataset import Dataset
from mindspore_gl.dataloader.samplers import RandomBatchSampler, DistributeRandomBatchSampler
from mindspore_gl.dataloader.dataloader import DataLoader


class MyDataset(Dataset):
    """
    Define user dataset using `Dataset`.
    """

    def __init__(self, start, end):
        assert end > start
        self.data = list(range(start, end))
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if isinstance(idx, list):
            value = []
            for i in idx:
                value.append(self.data[i])
            return value
        return self.data[idx]

    def __setitem__(self, key, value):
        self.data[key] = value


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dataset():
    """
    Feature: use `MyDataset` construct data
    Description: start = 0, end =12
    Expectation: success and equals [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].
    """
    dataset = MyDataset(0, 12)
    ret = np.array(list(dataset))
    expect = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert (ret - expect == 0).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_random_batch_sample():
    """
    Feature: use `RandomBatchSampler` sampling batch data from dataset
    Description: list(dataset) = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                 batch_size = 3
    Expectation: success and batch data as shape (4, 3).
    """
    dataset = list(range(0, 12))
    batch_size = 3
    sampler = RandomBatchSampler(dataset, batch_size)
    ret = list(sampler)
    np_ret = np.array(ret)
    assert np_ret.shape[0] == 4
    assert np_ret.shape[1] == 3


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_distribute_random_batch_sample():
    """
    Feature: use `DistributeRandomBatchSampler` sampling batch data from dataset on distribute devices
    Description: list(dataset) = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                 batch_size = 3
                 rank_id = 0
                 world_size = 2
    Expectation: success and batch data on one of two devices as shape (2, 3).
    """
    rank_id = 0
    world_size = 2
    batch_size = 3
    dataset = list(range(0, 12))
    dist_sampler = DistributeRandomBatchSampler(rank_id, world_size, dataset, batch_size)
    ret = list(dist_sampler)
    np_ret = np.array(ret)
    assert np_ret.shape[0] == 2
    assert np_ret.shape[1] == 3


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dataloader():
    """
    Feature: use `DataLoader` construct iterable batch data
    Description: list(dataset) = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                 sampler with batch_size = 3
                 collate_fn which map sample to 2*sample
    Expectation: success and map each sample to 2*sample.
    """
    batch_size = 3
    dataset = MyDataset(0, 12)
    sampler = RandomBatchSampler(dataset, batch_size)

    def collate_map(batch):
        data = []
        for sample in batch:
            data.append(sample*2)
        return data
    loader = DataLoader(dataset, sampler, collate_fn=collate_map)
    ret = list(loader)
    assert len(ret) == 4
    np_ret = np.array(ret)
    assert (np_ret % 2 == 0).all()
