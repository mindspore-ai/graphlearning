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
""" test graph ops """
import math
import scipy.sparse as sp
import numpy as np
import mindspore as ms
from mindspore_gl.dataset.imdb_binary import IMDBBinary
from mindspore_gl.graph import BatchHomoGraph, PadHomoGraph, PadMode, PadArray2d,\
    MindHomoGraph, get_laplacian, PadDirection, norm, UnBatchHomoGraph, remove_self_loop, add_self_loop,\
    gcn_norm, graph_csr_data, sampling_csr_data, batch_graph_csr_data, PadCsrEdge
import pytest

dataset = IMDBBinary("/home/workspace/mindspore_dataset/GNN_Dataset/")

nodes = list(range(7))
num_nodes = 7
src_idx = [0, 2, 2, 3, 4, 5, 5, 6]
dst_idx = [1, 0, 1, 5, 3, 4, 6, 4]


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_homograph():
    """
    Feature: test MindHomoGraph
    Description: test MindHomoGraph
    Expectation: Output result
    """
    graph = MindHomoGraph()
    graph.set_topo_coo([src_idx, dst_idx])
    adj_coo = np.array(graph.adj_coo)
    expect_output = np.array([[0, 2, 2, 3, 4, 5, 5, 6], [1, 0, 1, 5, 3, 4, 6, 4]])
    assert np.allclose(adj_coo, expect_output)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batch():
    """
    Feature: test MindHomoGraph
    Description: test MindHomoGraph
    Expectation: Output result
    """
    graphs = [dataset[0], dataset[1]]
    batch_graph_op = BatchHomoGraph()
    batch_graph = batch_graph_op(graphs)
    assert batch_graph.is_batched
    assert batch_graph[0].edge_count == graphs[0].edge_count
    assert batch_graph[0].node_count == graphs[0].node_count
    assert batch_graph[1].edge_count == graphs[1].edge_count
    assert batch_graph[1].node_count == graphs[1].node_count
    assert batch_graph.adj_coo.shape[1] == batch_graph.edge_count


def test_unbatch():
    """
    Feature: test UnBatchHomoGraph
    Description: test UnBatchHomoGraph
    Expectation: Output result
    """
    graphs = [dataset[0], dataset[1]]
    batch_graph_op = BatchHomoGraph()
    batch_graph = batch_graph_op(graphs)
    unbatch_fn = UnBatchHomoGraph()
    unbatch_graph = unbatch_fn(batch_graph)
    assert unbatch_graph[0].edge_count == graphs[0].edge_count
    assert unbatch_graph[0].node_count == graphs[0].node_count
    assert unbatch_graph[1].edge_count == graphs[1].edge_count
    assert unbatch_graph[1].node_count == graphs[1].node_count

def test_pad_array2d():
    """
    Feature: test PadArray2d
    Description: test PadArray2d
    Expectation: Output result
    """
    pad_op = PadArray2d(dtype=np.float32, mode=PadMode.CONST, direction=PadDirection.COL, size=(10, 2), fill_value=0)
    node_list = np.array([[1, 2], [2, 4]])
    pad_res = pad_op(node_list)
    expect_output = np.array([[1., 2.], [2., 4.], [0., 0.], [0., 0.], [0., 0.],
                              [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]])
    assert np.allclose(pad_res, expect_output)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batchmeta():
    """
    Feature: test BatchMeta
    Description: test BatchMeta
    Expectation: Output result
    """
    graphs = [dataset[0], dataset[1]]
    batch_graph_op = BatchHomoGraph()
    batch_graph = batch_graph_op(graphs)
    assert batch_graph.batch_meta[0] == (dataset.graph_nodes[1], dataset.graph_edges[1])


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pad_const():
    """
    Feature: test PadMode.CONST
    Description: test PadMode.CONST
    Expectation: Output result
    """
    graph = dataset[0]
    n_node = graph.node_count + 1
    n_edge = graph.edge_count + 30
    pad_graph_op = PadHomoGraph(mode=PadMode.CONST, n_node=n_node, n_edge=n_edge)
    pad_res = pad_graph_op(graph)
    assert pad_res.edge_count == n_edge
    assert pad_res.node_count == n_node
    assert pad_res.is_batched
    assert pad_res[0].edge_count == graph.edge_count
    assert pad_res[0].node_count == graph.node_count
    assert pad_res[1].edge_count == 30
    assert pad_res[1].node_count == 1
    assert pad_res.adj_coo.shape[1] == pad_res.edge_count


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pad_auto():
    """
    Feature: test PadMode.AUTO
    Description: test PadMode.AUTO
    Expectation: Output result
    """
    graph = dataset[0]
    pad_graph_op = PadHomoGraph(mode=PadMode.AUTO)
    pad_res = pad_graph_op(graph)
    assert pad_res.edge_count == 1 << math.ceil(math.log2(graph.edge_count))
    assert pad_res.node_count == 1 << math.ceil(math.log2(graph.node_count))
    assert pad_res.is_batched
    assert pad_res[0].edge_count == graph.edge_count
    assert pad_res[0].node_count == graph.node_count
    assert pad_res.adj_coo.shape[1] == pad_res.edge_count


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batch_and_pad_const():
    """
    Feature: test batch and pad const
    Description: test batch and pad const
    Expectation: Output result
    """
    graphs = [dataset[0], dataset[1]]
    batch_graph_op = BatchHomoGraph()
    batch_graph = batch_graph_op(graphs)
    n_node = batch_graph.node_count + 1
    n_edge = batch_graph.edge_count + 30
    pad_graph_op = PadHomoGraph(mode=PadMode.CONST, n_node=n_node, n_edge=n_edge)
    pad_res = pad_graph_op(batch_graph)
    assert pad_res.edge_count == n_edge
    assert pad_res.node_count == n_node
    assert pad_res.is_batched
    assert pad_res[0].edge_count == graphs[0].edge_count
    assert pad_res[0].node_count == graphs[0].node_count
    assert pad_res[1].edge_count == graphs[1].edge_count
    assert pad_res[1].node_count == graphs[1].node_count
    assert pad_res[2].edge_count == 30
    assert pad_res[2].node_count == 1
    assert pad_res.adj_coo.shape[1] == pad_res.edge_count


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batch_and_pad_auto():
    """
    Feature: test batch and pad auto
    Description: test batch and pad auto
    Expectation: Output result
    """
    graphs = [dataset[0], dataset[1]]
    batch_graph_op = BatchHomoGraph()
    batch_graph = batch_graph_op(graphs)
    pad_graph_op = PadHomoGraph(mode=PadMode.AUTO)
    pad_res = pad_graph_op(batch_graph)
    assert pad_res.edge_count == 1 << math.ceil(math.log2(batch_graph.edge_count))
    assert pad_res.node_count == 1 << math.ceil(math.log2(batch_graph.node_count))
    assert pad_res.is_batched
    assert pad_res[0].edge_count == graphs[0].edge_count
    assert pad_res[0].node_count == graphs[0].node_count
    assert pad_res[1].edge_count == graphs[1].edge_count
    assert pad_res[1].node_count == graphs[1].node_count
    assert pad_res.adj_coo.shape[1] == pad_res.edge_count

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_laplacian():
    """
    Feature: test get_laplacian
    Description: test get_laplacian
    Expectation: Output result
    """
    edge_index = [src_idx, dst_idx]
    edge_index = ms.Tensor(edge_index, ms.int32)
    edge_weight = ms.Tensor([1, 2, 1, 2, 1, 2, 1, 2], ms.float32)
    _, edge_weight = get_laplacian(edge_index, num_nodes, edge_weight, 'sym')
    expect_output = np.array([-0., -1.1547005, -0., -0.8164965, -0.7071067, -1.1547005, -0.40824825,
                              -1.4142134, 1., 1., 1., 1., 1., 1., 1.])
    assert np.allclose(edge_weight.asnumpy(), expect_output)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_norm():
    """
    Feature: test norm
    Description: test norm
    Expectation: Output result
    """
    edge_index = [src_idx, dst_idx]
    edge_index = ms.Tensor(edge_index, ms.int32)
    _, edge_weight = norm(edge_index, num_nodes)
    expect_output = np.array([-0., -0.7071067, -0., -0.7071067, -1., -0.7071067, -0.7071067,
                              -1., 1., 1., 1., 1., 1., 1., 1.])
    assert np.allclose(edge_weight.asnumpy(), expect_output)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_remove_loop():
    """
    Feature: Test that the removal of self-loops is performed correctly.
    Description: Test remove self loop
    Expectation: Output result
    """

    adj = sp.csr_matrix(([1, 1, 1, 1], ([0, 1, 2, 3], [0, 1, 2, 3])), shape=(4, 4)).tocoo()
    adj_new = remove_self_loop(adj, mode='dense')
    for i in range(3):
        assert adj_new[i][i] == 0

    adj = sp.csr_matrix(([1, 2, 3, 4], ([0, 1, 2, 2], [0, 1, 2, 1])), shape=(3, 3)).tocoo()
    adj = remove_self_loop(adj, 'coo')
    assert ~adj.diagonal().any()

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_add_loop():
    """
    Feature: Test that adding a self-loop is executed correctly.
    Description: Test add self loop
    Expectation: Output result
    """

    edge_index = [src_idx, dst_idx]
    edge_index = ms.Tensor(edge_index, ms.int32)
    edge_weight = ms.Tensor([1, 2, 1, 2, 1, 2, 1, 2], ms.float32)
    node = 7
    fill_value = ms.Tensor([1] * node, ms.float32)
    new_adj = add_self_loop(edge_index, edge_weight, node, fill_value, mode='dense')
    for i in range(node):
        assert new_adj[i][i] != 0

    edge_index, edge_weight = add_self_loop(edge_index, edge_weight, node, fill_value, mode='coo')
    count = 0
    for i in range(edge_index.shape[1]):
        if edge_index[0, i] == edge_index[1, i]:
            count += 1
    assert count >= node

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gcn_norm():
    """
    Feature: test gcn_norm
    Description: test gcn_norm
    Expectation: Output result
    """
    edge_index = [src_idx, dst_idx]
    edge_index = ms.Tensor(edge_index, ms.int32)
    _, edge_weight = gcn_norm(edge_index, num_nodes)
    expect_output = np.array([[0.40824825], [0.7071067], [0.57735026], [0.4999999], [0.40824825],
                              [0.40824825], [0.4999999], [0.40824825], [0.4999999], [0.3333333],
                              [1.], [0.4999999], [0.3333333], [0.4999999], [0.4999999]]
                             )
    assert np.allclose(edge_weight.asnumpy(), expect_output)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_csr_graph():
    """
    Feature: test csr whole graph
    Description: test csr whole graph
    Expectation: Output result
    """
    node_feat = np.array([[1, 2, 3, 4], [2, 4, 1, 3], [1, 3, 2, 4],
                          [9, 7, 5, 8], [8, 7, 6, 5], [8, 6, 4, 6], [1, 2, 1, 1]], np.float32)
    n_nodes = 7
    n_edges = 8
    src_idx_array = np.array(src_idx, np.int32)
    dst_idx_array = np.array(dst_idx, np.int32)
    node_label = np.array([0, 1, 0, 1, 0, 1, 0])
    train_mask = np.array([True, True, True, True, False, False, False])
    val_mask = np.array([False, False, False, False, True, True, True])
    g, _, _, node_feat, node_label,\
    train_mask, val_mask, _ = graph_csr_data(src_idx_array, dst_idx_array, n_nodes, n_edges, node_feat, node_label,
                                             train_mask, val_mask, test_mask=None, rerank=True)
    expect_indices = np.array([2, 3, 5, 6, 3, 4, 0, 6])
    expect_indptr = np.array([0, 2, 4, 5, 6, 7, 8, 8])
    expect_node_feat = np.array([[8., 7., 6., 5.],
                                 [2., 4., 1., 3.],
                                 [1., 2., 1., 1.],
                                 [8., 6., 4., 6.],
                                 [9., 7., 5., 8.],
                                 [1., 2., 3., 4.],
                                 [1., 3., 2., 4.]])
    expect_node_label = np.array([0, 1, 0, 1, 1, 0, 0])
    assert np.allclose(g[0].asnumpy(), expect_indices)
    assert np.allclose(g[1].asnumpy(), expect_indptr)
    assert np.allclose(expect_node_feat, node_feat)
    assert np.allclose(expect_node_label, node_label)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_csr_sample_graph():
    """
    Feature: test csr sampled graph
    Description: test csr sampled graph
    Expectation: Output result
    """
    node_feat = np.array([[1, 2, 3, 4], [2, 4, 1, 3], [1, 3, 2, 4],
                          [9, 7, 5, 8], [8, 7, 6, 5], [8, 6, 4, 6], [1, 2, 1, 1]], np.float32)
    n_nodes = 7
    n_edges = 8
    src_idx_array = np.array(src_idx, np.int32)
    dst_idx_array = np.array(dst_idx, np.int32)
    seeds_idx = np.array([0, 3, 5])
    g, seeds_idx, node_feat = sampling_csr_data(src_idx_array, dst_idx_array, n_nodes, n_edges,
                                                seeds_idx, node_feat, rerank=True)
    expect_indices = np.array([2, 3, 5, 6, 3, 4, 0, 6])
    expect_indptr = np.array([0, 2, 4, 5, 6, 7, 8, 8])
    expect_seeds_idx = np.array([5, 4, 3])
    expect_node_feat = np.array([[8., 7., 6., 5.],
                                 [2., 4., 1., 3.],
                                 [1., 2., 1., 1.],
                                 [8., 6., 4., 6.],
                                 [9., 7., 5., 8.],
                                 [1., 2., 3., 4.],
                                 [1., 3., 2., 4.]])
    assert np.allclose(g[0].asnumpy(), expect_indices)
    assert np.allclose(g[1].asnumpy(), expect_indptr)
    assert np.allclose(expect_seeds_idx, seeds_idx)
    assert np.allclose(expect_node_feat, node_feat)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_csr_batched_graph():
    """
    Feature: test csr batched graph
    Description: test csr batched graph
    Expectation: Output result
    """
    node_feat = np.array([[1, 2, 3, 4], [2, 4, 1, 3], [1, 3, 2, 4],
                          [9, 7, 5, 8], [8, 7, 6, 5], [8, 6, 4, 6], [1, 2, 1, 1]], np.float32)
    n_nodes = 7
    n_edges = 8
    src_idx_array = np.array(src_idx, np.int32)
    dst_idx_array = np.array(dst_idx, np.int32)
    node_map_idx = np.array([0, 0, 0, 0, 1, 1, 1])
    g, node_map_idx, node_feat = batch_graph_csr_data(src_idx_array, dst_idx_array,
                                                      n_nodes, n_edges, node_map_idx, node_feat, rerank=True)
    expect_node_map_idx = np.array([1, 0, 1, 1, 0, 0, 0])
    expect_indices = np.array([2, 3, 5, 6, 3, 4, 0, 6])
    expect_indptr = np.array([0, 2, 4, 5, 6, 7, 8, 8])
    expect_node_feat = np.array([[8., 7., 6., 5.],
                                 [2., 4., 1., 3.],
                                 [1., 2., 1., 1.],
                                 [8., 6., 4., 6.],
                                 [9., 7., 5., 8.],
                                 [1., 2., 3., 4.],
                                 [1., 3., 2., 4.]])
    assert np.allclose(g[0].asnumpy(), expect_indices)
    assert np.allclose(g[1].asnumpy(), expect_indptr)
    assert np.allclose(expect_node_map_idx, node_map_idx)
    assert np.allclose(expect_node_feat, node_feat)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pad_csr_edges():
    """
    Feature: test csr batched graph
    Description: test csr batched graph
    Expectation: Output result
    """
    node_pad = 10
    origin_edge_index = np.array([[0, 1, 2, 4],
                                  [2, 3, 1, 1]])
    pad_length = 20
    pad_op = PadCsrEdge(node_pad, length=pad_length, mode=PadMode.CONST)
    res = pad_op(origin_edge_index)
    expected_padding = np.array([[0, 1, 2, 4, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8],
                                 [2, 3, 1, 1, 5, 6, 7, 8, 6, 7, 8, 5, 7, 8, 5, 6, 8, 5, 6, 7]])
    assert np.allclose(expected_padding, res)
