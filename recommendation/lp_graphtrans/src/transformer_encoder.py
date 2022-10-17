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
"""transformer encoder"""
import argparse
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor


class TransformerNodeEncoderMS(ms.nn.Cell):
    """
        TransformerNodeEncoderMS, implement transformer's encoder layer.
    """

    def __init__(self, args):
        super().__init__()
        self.model_dim = args.model_dim
        self.num_layer = args.count_trans_encoder

        self.transformer = ms.nn.transformer.TransformerEncoder(
            batch_size=args.batch_size,
            num_layers=args.count_trans_encoder,
            hidden_size=args.model_dim,
            ffn_hidden_size=args.dim_feedforward,
            seq_length=args.seq_length + 1,
            num_heads=args.nhead,
            attention_dropout_rate=args.transformer_dropout,
            hidden_dropout_rate=args.transformer_dropout,
            hidden_act=args.transformer_activation
        )
        self.max_input_len = args.max_input_len

        self.norm_input = None
        if args.transformer_norm_input:
            self.norm_input = ms.nn.LayerNorm((args.model_dim,))

        self.cls_embedding = None
        if args.gnn_pool_type == "cls":
            self.cls_embedding = \
                ms.Parameter(Tensor(np.random.random((1, 1, args.model_dim)),
                                    ms.float32), requires_grad=True)

    def construct(self, padded_h_node, src_padding_mask):
        """
        padded_h_node: n_b x B x h_d
        src_key_padding_mask: B x n_b
        """

        src_padding_mask = src_padding_mask.astype(ms.float32)
        if self.cls_embedding is not None:
            broadcast_to_e = \
                ops.BroadcastTo((1, padded_h_node.shape[1], -1))
            expand_cls_embedding = broadcast_to_e(self.cls_embedding)

            concat_op = ops.Concat(0)
            padded_h_node = \
                concat_op([padded_h_node, expand_cls_embedding])

            zeros_fn = ops.Zeros()
            zeros = \
                zeros_fn((src_padding_mask.shape[0], 1), ms.float32)
            concat_op = ops.Concat(1)
            src_padding_mask = concat_op([src_padding_mask, zeros])

        if self.norm_input is not None:
            padded_h_node = self.norm_input(padded_h_node)

        padded_h_node = ops.transpose(padded_h_node, (1, 0, 2))
        src_padding_mask = ops.expand_dims(src_padding_mask, -1)
        broadcast_to_m = \
            ops.BroadcastTo((-1, -1, src_padding_mask.shape[1]))
        src_padding_mask = broadcast_to_m(src_padding_mask)

        transformer_out, _ = \
            self.transformer(padded_h_node, src_padding_mask)
        transformer_out = ops.transpose(transformer_out, (1, 0, 2))

        return transformer_out, src_padding_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test.")
    parser.add_argument("--model_dim", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--transformer_dropout", type=float, default=0.3)
    parser.add_argument("--transformer_activation", type=str, default="relu")
    parser.add_argument("--count_trans_encoder", type=int, default=4)
    parser.add_argument("--max_input_len", default=1000)
    parser.add_argument("--transformer_norm_input",
                        action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seq_length", type=int, default=3)
    parser.add_argument("--gnn_pool_type", type=str, default="cls")
    args_demo = parser.parse_args()

    padded_h_node_demo = \
        ms.Tensor(np.random.random((3, 256, 128)),
                  dtype=ms.float32)
    src_key_padding_mask_demo = \
        ms.Tensor(np.ones((256, 3)), dtype=ms.int32)
    src_key_padding_mask_demo = \
        src_key_padding_mask_demo.astype(ms.bool_)
    dome_net = TransformerNodeEncoderMS(args_demo)

    res, _ = \
        dome_net(padded_h_node_demo, src_key_padding_mask_demo)
    print(res.shape)
