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

"""Reading and building interface for graph datasets."""
from .cora import CoraV2
from .reddit import Reddit
from .mag240m import MAG240MDataset
from .imdb_binary import IMDBBinary
from .alchemy import Alchemy
from .enzymes import Enzymes
from .blog_catalog import BlogCatalog
from .ppi import PPI

__all__ = [
    "CoraV2",
    "Reddit",
    "IMDBBinary",
    "PPI",
    "BlogCatalog",
    "Alchemy",
    "Enzymes"
]
