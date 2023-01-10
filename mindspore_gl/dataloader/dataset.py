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
"""
Mappable dataset.
"""
from typing import (
    Generic,
    TypeVar,
)

Tco = TypeVar('Tco', covariant=True)


class Dataset(Generic[Tco]):
    r"""
    Mappable Dataset Definition, an abstract class represent Dataset.
    All datasets should subclass it which represent a map relation from key to sample.
    All subclass should overwrite `__getitem__`, which implement fetch a sample given a key.

    Note:
        :class:`mindspore_gl.dataloader.Dataloader` needs a `Dataset` instance as input. It is mutually exclusive
        with `Sampler` which yields indices.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindspore_gl.dataloader import Dataset
        >>> class MyDataset(Dataset):
        >>>    def __init__(self, *args, **kwargs):
        >>>         ...
        >>> my_dataset = MyDataset()
    """
    def __getitem__(self, index):
        raise NotImplementedError
