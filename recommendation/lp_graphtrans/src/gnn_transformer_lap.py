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
"""gnn transformer lap"""
import math

import mindspore as ms
import mindspore.ops as ops
from mindspore_gl import BatchedGraph
from mindspore_gl.nn import GNNCell

from recommendation.lp_graphtrains.utils import pad_batch_ms
from transformer_encoder import TransformerNodeEncoderMS
from gnn_module import gnn_node_embedding


class GNNTransformerMS(GNNCell):
    """
        GNNTransformerMS, a network of both gnn and transformer
    """

    def __init__(self, tasks_count, point_encoder, border_encoder, args):
        super().__init__()
        self.gnn_node_network = gnn_node_embedding(
            args.gnn_virtual_flag,
            args.gnn_layer_count,
            args.gnn_embedding,
            point_encoder,
            border_encoder,
            jk=args.gnn_JK,
            drop_ratio=args.gnn_dropout_value,
            residual=args.gnn_residual,
            gnn_type=args.gnn_type,
        )
        self.stop_gnn = args.stop_gnn

        gnn_embedding = \
            2 * args.gnn_embedding if args.gnn_JK == \
                                    "cat" else args.gnn_embedding
        self.gnn_to_trans = ms.nn.Dense(gnn_embedding, args.model_dim)

        self.use_lap = args.use_lap
        if self.use_lap == 'tf' or self.use_lap == 'all':
            self.tf_pos_encoder_embedding = \
                ms.nn.Dense(args.lap_dim, args.model_dim)
        if self.use_lap == 'gnn' or self.use_lap == 'all':
            self.gnn_pos_encoder_embedding = \
                ms.nn.Dense(args.lap_dim, args.gnn_embedding)

        self.trans_encoder_network = TransformerNodeEncoderMS(args)
        self.count_trans_encoder = args.count_trans_encoder  # 4
        self.count_trans_encoder_masked = args.count_trans_encoder_masked  # 0
        self.tasks_count = tasks_count  # 词表大小
        self.pool_type = args.gnn_pool_type  # mean
        self.gnn_pred_dense_list = ms.nn.CellList()
        self.max_input_len = args.max_input_len

        self.max_sequence = args.max_sequence
        output_dim = args.model_dim

        if args.max_sequence is None:
            self.graph_pred_linear = \
                ms.nn.Dense(output_dim, self.tasks_count)
        else:
            for _ in range(args.max_sequence):
                self.gnn_pred_dense_list. \
                    append(ms.nn.Dense(output_dim, self.tasks_count))

    def construct(self, batched_data, perturb, bg: BatchedGraph):
        """gnn transformer lap forward"""
        if self.use_lap == 'gnn' or self.use_lap == 'all':
            lap_pos_enc = self.batch_lap_encoding(batched_data)
            lap_pos_enc_embed = \
                self.gnn_pos_encoder_embedding(lap_pos_enc)

            perturb = lap_pos_enc_embed
        h_node = self.gnn_node_network(batched_data, perturb, bg)  # GNN得到的表征
        h_node = self.gnn_to_trans(h_node)

        if self.use_lap == 'tf' or self.use_lap == 'all':
            lap_pos_enc = self.batch_lap_encoding(batched_data)
            lap_pos_enc_embed = self.tf_pos_encoder_embedding(lap_pos_enc)
            h_node += lap_pos_enc_embed
        padded_h_node, src_padding_mask = pad_batch_ms(h_node, batched_data[3],
                                                       self.max_input_len)
        trans_output = padded_h_node

        if self.count_trans_encoder > 0:
            trans_output = trans_output.astype(ms.float32)
            trans_output, _ = self.trans_encoder_network(trans_output,
                                                         src_padding_mask)

        if self.pool_type in ["last", "cls"]:
            hidden_gnn = trans_output[-1]
        elif self.pool_type == "mean":
            hidden_gnn = trans_output.sum(
                0) / src_padding_mask.sum(-1, keepdim=True)
        else:
            raise NotImplementedError

        if self.max_sequence is None:
            out = self.graph_pred_linear(hidden_gnn)
            return out
        predict_list = []
        for i in range(self.max_sequence):
            predict_list.append(self.gnn_pred_dense_list[i](hidden_gnn))
        return predict_list

    # notehere
    def batch_lap_encoding(self, batched_data):
        batch_lap_pos_enc = batched_data.lap_list  # [batch,N,dim]
        sign_flip = ms.ops.UniformReal(batch_lap_pos_enc.size(-1))
        sign_flip[sign_flip >= 0.5] = 1.0
        sign_flip[sign_flip < 0.5] = -1.0
        batch_lap_pos_enc = batch_lap_pos_enc * \
                            sign_flip.unsqueeze(0)  # 按列相乘 N*8

        return batch_lap_pos_enc


class PositionalEncodingMS(ms.nn.Cell):
    """position encoding"""
    def __init__(self, model_dim: int, dropout: float = 0.9, max_len: int = 5000):
        super().__init__()
        self.dropout = ms.nn.Dropout(p=dropout)
        position = ms.numpy.arange(max_len)
        position = ops.expand_dims(position, 1)

        div_term = \
            ms.ops.Exp(ms.numpy.arange(0, model_dim, 2)
                       * (-math.log(10000.0) / model_dim))
        pe = ms.ops.Zeros(max_len, 1, model_dim)
        pe[:, 0, 0::2] = ms.ops.Sin(position * div_term)
        pe[:, 0, 1::2] = ms.ops.Cos(position * div_term)
        self.register_buffer('pe', pe)

    def construct(self, input_data):
        """
        Args:
            input_data: Tensor, shape :math:`[seq_len, batch_size, embedding_dim]`
        """
        input_data = input_data + self.pe[:input_data.size(0)]
        return self.dropout(input_data)
