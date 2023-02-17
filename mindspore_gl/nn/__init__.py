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
"""APIs for graph convolutions."""
from .gnn_cell import GNNCell
from .conv import (
    AGNNConv,
    APPNPConv,
    CFConv,
    ChebConv,
    DOTGATConv,
    EDGEConv,
    GCNConv,
    EGConv,
    GATConv,
    GatedGraphConv,
    GATv2Conv,
    GCNConv2,
    GINConv,
    GMMConv,
    MeanConv,
    NNConv,
    SAGEConv,
    SGConv,
    TAGConv,
    GCNEConv
)
from .temporal import (
    ASTGCN, STConv
)
from .glob import (
    AvgPooling,
    GlobalAttentionPooling,
    MaxPooling,
    SAGPooling,
    Set2Set,
    SortPooling,
    SumPooling,
    WeightAndSum
)

__all__ = [
    'GNNCell',
    'AGNNConv',
    'APPNPConv',
    'CFConv',
    'ChebConv',
    'DOTGATConv',
    'EDGEConv',
    'EGConv',
    'GATConv',
    'GatedGraphConv',
    'GATv2Conv',
    'GCNConv',
    'ASTGCN',
    'STConv',
    'GINConv',
    'GMMConv',
    'NNConv',
    'SAGEConv',
    'SGConv',
    'TAGConv',
    'GCNConv2',
    'MeanConv',
    'AvgPooling',
    'GlobalAttentionPooling',
    'MaxPooling',
    'SAGPooling',
    'Set2Set',
    'SortPooling',
    'SumPooling',
    'WeightAndSum',
    'GCNEConv'
]
__all__.sort()
