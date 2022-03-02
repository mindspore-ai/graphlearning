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
"""test k_hop_subgraph"""
import pytest
import numpy as np
from mindspore_gl.sampling.k_hop_sampling import k_hop_subgraph
from mindspore_gl.graph.graph import MindHomoGraph


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_k_hop_subgraph():
    """Feature: test K-hop subgraph sampling
    Description: sampling a 2-hop subgraph
    res["subset"]: np.array([0, 1, 2, 3, 4])
    res["adj_coo"]: np.array([[0, 1, 1, 2, 3, 0, 4, 4], [1, 0, 2, 1, 0, 3, 4, 3]])
    res["inv"]: [0, 3]
    res["edge_mask"]: [True, True, True, True, True, True, True, True, False, False]
    Expectation: results == {"subset":subset, "adj_coo":adj_coo, "inv":inv, "edge_mask":edge_mask}
    """
    expected_res = {"subgraph_subset": np.array([0, 1, 2, 3, 4]),
                    "subgraph_adj_coo": np.array([[0, 1, 1, 2, 3, 0, 3, 4], [1, 0, 2, 1, 0, 3, 4, 3]]),
                    "inv": [0, 3],
                    "edge_mask": [True, True, True, True, True, True, True, True, False, False]}
    graph = MindHomoGraph()
    coo_array = np.array([[0, 1, 1, 2, 3, 0, 3, 4, 2, 5],
                          [1, 0, 2, 1, 0, 3, 4, 3, 5, 2]])
    graph.set_topo_coo(coo_array)
    graph.node_count = 6
    graph.edge_count = 10

    res = k_hop_subgraph([0, 3], 2, graph.adj_coo, graph.node_count, relabel_nodes=True)

    assert (res["subset"] == expected_res["subgraph_subset"]).all()
    assert (res["adj_coo"] == expected_res["subgraph_adj_coo"]).all()
    assert (res["inv"] == expected_res["inv"]).all()
    assert (res["edge_mask"] == expected_res["edge_mask"]).all()
