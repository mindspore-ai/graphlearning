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
"""train laplace graph transformer networks"""
import argparse
import numpy as np

import mindspore as ms
from mindspore_gl import BatchedGraphField

from src.gnn_transformer_lap import GNNTransformerMS


def main(args):
    """train lgtn"""
    x = ms.Tensor(np.random.random((7, 4)), dtype=ms.float32)
    edge_index = \
        ms.Tensor([[0, 2, 2, 3, 4, 5, 5, 6],
                   [1, 0, 1, 5, 3, 4, 6, 4]], ms.int32)
    edge_attr = ms.Tensor(np.random.random((8, 4)), dtype=ms.float32)
    batch = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)

    ### batch graph
    n_nodes = x.shape[0]
    n_edges = edge_attr.shape[0]
    src_idx = edge_index[0, :]
    dst_idx = edge_index[1, :]
    ver_subgraph_idx = batch

    batch_np = batch.asnumpy()
    edge_index_np = edge_index.asnumpy()
    edge_batch = [batch_np[idx] for idx in edge_index_np[0]]
    edge_subgraph_idx = ms.Tensor(edge_batch)
    graph_mask = ms.Tensor(np.ones((1, 2)), dtype=ms.int32)
    edge_weight = ms.Tensor(np.random.random((8, 4)), dtype=ms.float32)
    batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes,
                                            n_edges, ver_subgraph_idx,
                                            edge_subgraph_idx, graph_mask)

    batch_data_ms = (x, edge_index, edge_attr,
                     batch, edge_weight, None)
    model_ms = \
        GNNTransformerMS(16, None, None, args)
    res_ms = \
        model_ms(batch_data_ms, None, *batched_graph_field.get_batched_graph())

    print(res_ms)

if __name__ == '__main__':
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")

    parser = argparse.ArgumentParser(description="test")
    parser.add_argument('--wandb_run_idx', type=str, default=None)
    parser.add_argument('--dataset', type=str, default="ogbg-code",
                        help='dataset name (default: ogbg-code)')
    parser.add_argument('--aug', type=str, default='baseline')
    parser.add_argument('--max_sequence', type=int, default=None)
    parser.add_argument('--model_type', type=str, default='gnn-transformer')
    parser.add_argument('--gnn_pool_type', type=str, default='cls')
    parser.add_argument('--gnn_type', type=str, default='gin')
    parser.add_argument('--gnn_virtual_flag', action='store_true', default=True)
    parser.add_argument('--gnn_dropout_value', type=float, default=0.7)
    parser.add_argument('--gnn_layer_count', type=int, default=3)
    parser.add_argument('--gnn_embedding', type=int, default=4)
    parser.add_argument('--gnn_JK', type=str, default='cat')
    parser.add_argument('--gnn_residual', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default="0")
    parser.add_argument('--eval_batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--scheduler', type=str, default=None)
    parser.add_argument('--pct_start', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--grad_clip', type=float, default=None)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max_lr', type=float, default=0.001)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--test-freq', type=int, default=1)
    parser.add_argument('--start-eval', type=int, default=15)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--use_lap', type=str, default=None)
    parser.add_argument('--lap_dim', type=int, default=None)
    parser.add_argument("--pretrained_gnn", type=str, default=None)
    parser.add_argument("--stop_gnn", type=int, default=None)
    parser.add_argument("--model_dim", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--transformer_dropout", type=float, default=0.7)
    parser.add_argument("--transformer_activation", type=str, default="relu")
    parser.add_argument("--count_trans_encoder", type=int, default=4)
    parser.add_argument("--max_input_len", default=1000)
    parser.add_argument("--transformer_norm_input",
                        action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_length", type=int, default=4)
    parser.add_argument("--count_trans_encoder_masked", type=int, default=0)
    args_demo = parser.parse_args()
    main(args_demo)
