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
from mindspore_gl.dataset import IMDBBinary
from mindspore_gl.graph.ops import BatchHomoGraph, PadHomoGraph, PadMode
import pytest

dataset = IMDBBinary("/home/workspace/mindspore_dataset/GNN_Dataset/")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batch():
    """ test batch """
    graphs = [dataset[0], dataset[1]]
    batch_graph_op = BatchHomoGraph()
    batch_graph = batch_graph_op(graphs)
    assert batch_graph.is_batched
    assert batch_graph[0].edge_count == graphs[0].edge_count
    assert batch_graph[0].node_count == graphs[0].node_count
    assert batch_graph[1].edge_count == graphs[1].edge_count
    assert batch_graph[1].node_count == graphs[1].node_count
    assert batch_graph.adj_coo.shape[1] == batch_graph.edge_count


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pad_const():
    """ test pad const """
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
    """ test pad auto """
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
    """ test batch and pad const """
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
    """ test batch and pad auto """
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
