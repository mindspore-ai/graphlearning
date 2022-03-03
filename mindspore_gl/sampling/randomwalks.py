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

"""random walks on graphs"""
import numpy as np
from mindspore_gl.graph import MindHomoGraph
from mindspore_gl import sample_kernel

__all__ = ['random_walk_unbias_on_homo']


def random_walk_unbias_on_homo(homo_graph: MindHomoGraph,
                               seeds: np.ndarray,
                               walk_length: int,
                               default_node: int = -1):
    """
    random walks on homo graph

    Args:
        homo_graph(MindHomoGraph): the source graph which is sampled from
        seeds(np.ndarray) : random seeds for sampling
        walk_length(int): sample path length
        default_node(int): node index which the random walk traces start
    """
    default_node = int(default_node)
    # sample
    out = sample_kernel.random_walk_cpu_unbias(homo_graph.adj_csr.indptr,
                                               homo_graph.adj_csr.indices,
                                               walk_length, seeds, default_node)
    return out
