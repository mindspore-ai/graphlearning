"""test k_hop_subgraph"""
import pytest
import numpy as np
from mindspore_gl.sampling.k_hop_sampling import gen_edge_lookup, fast_k_hop_subgraph, k_hop_subgraph
from mindspore_gl.graph import MindHomoGraph


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_fast_k_hop_subgraph():
    """Feature: test fast K-hop subgraph sampling
    Description: sampling a 2-hop subgraph, . The returned subgraph will be slightly different from k_hop_subgraph()
    that edges connecting the subgraph's nodes but exceed the hopping distance won't be returned. Therefore, edge 2
    doesn't exists in the subgraph.
    res["subset"]: np.array([0, 1, 2, 3, 4])
    res["subgraph_edge_subset"]: np.array([0, 1, 3, 4, 5, 6, 7])
    res["adj_coo"]: np.array([[0, 1, 2, 3, 0, 4, 4], [1, 0, 1, 0, 3, 4, 3]])
    res["inv"]: [0, 3]
    Expectation: results == {"subset":subset, "adj_coo":adj_coo, "inv":inv, "edge_mask":edge_mask}
    """
    expected_res = {"subgraph_subset": np.array([0, 1, 2, 3, 4]),
                    "subgraph_edge_subset": np.array([0, 1, 3, 4, 5, 6, 7]),
                    "subgraph_adj_coo": np.array([[0, 1, 2, 3, 0, 3, 4], [1, 0, 1, 0, 3, 4, 3]]),
                    "inv": [0, 3]}
    graph = MindHomoGraph()
    coo_array = np.array([[0, 1, 1, 2, 3, 0, 3, 4, 2, 5],
                          [1, 0, 2, 1, 0, 3, 4, 3, 5, 2]])
    graph.set_topo_coo(coo_array)
    graph.node_count = 6
    graph.edge_count = 10

    lookup = gen_edge_lookup(graph.adj_coo, graph.node_count)
    res = fast_k_hop_subgraph(lookup, [0, 3], 2, graph.adj_coo, relabel_nodes=True)

    assert (res["subset"] == expected_res["subgraph_subset"]).all()
    assert (res["edge_subset"] == expected_res["subgraph_edge_subset"]).all()
    assert (res["adj_coo"] == expected_res["subgraph_adj_coo"]).all()
    assert (res["inv"] == expected_res["inv"]).all()


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
