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
"""HAN"""
from typing import List
import mindspore as ms
from mindspore_gl.nn.conv import GATConv
from mindspore_gl import GNNCell
from mindspore_gl import HeterGraph


class SemanticAttention(ms.nn.Cell):
    """Semantic Attention class"""
    def __init__(self,
                 in_feat_size: int,
                 hidden_size: int = 128) -> None:
        super().__init__()
        self.proj = ms.nn.SequentialCell(
            ms.nn.Dense(in_feat_size, hidden_size),
            ms.nn.Tanh(),
            ms.nn.Dense(hidden_size, 1, has_bias=False)
        )

    def construct(self, x):
        """construct function"""
        h = ms.ops.ReduceMean()(self.proj(x), 0)
        beta = ms.ops.Softmax(0)(h)
        beta = ms.ops.BroadcastTo((ms.ops.Shape()(x)[0],) + ms.ops.Shape()(beta))(beta)
        return ms.ops.ReduceSum()(beta * x, 1)


class HANLayer(GNNCell):
    """HAN Layer"""
    def __init__(self,
                 num_meta_paths: int,
                 in_feat_size: int,
                 out_size: int,
                 num_heads: int,
                 dropout: float) -> None:
        super().__init__()
        gats = []
        print("in_feat size", in_feat_size)
        for _ in range(num_meta_paths):
            gats.append(GATConv(in_feat_size, out_size, num_heads, dropout, dropout, activation=ms.nn.ELU()))

        self.gats = ms.nn.CellList(gats)
        self.semantic = SemanticAttention(out_size * num_heads)
        self.num_meta_paths = num_meta_paths

    def construct(self, h, hg: HeterGraph):
        """construct function"""
        semantic_embeddings = []
        for i in range(self.num_meta_paths):
            semantic_embeddings.append(self.gats[i](h, *hg.get_homo_graph(i)))

        semantic_embeddings = ms.ops.Stack(1)(semantic_embeddings)
        ret = self.semantic(semantic_embeddings)
        return ret


class HAN(GNNCell):
    """HAN"""
    def __init__(self,
                 num_meta_paths: int,
                 in_feat_size: int,
                 hidden_size: int,
                 out_size: int,
                 num_heads: List[int],
                 dropout: float
                 ) -> None:
        super().__init__()
        layers = [HANLayer(num_meta_paths, in_feat_size, hidden_size, num_heads[0], dropout)]
        for i in range(1, len(num_heads)):
            layers.append(HANLayer(num_meta_paths, hidden_size * num_heads[i - 1], hidden_size, num_heads[i], dropout))
        self.layers = ms.nn.CellList(layers)
        self.predict = ms.nn.Dense(hidden_size * num_heads[-1], out_size)

    def construct(self, h, hg: HeterGraph):
        """construct function"""
        for conv in self.layers:
            h = conv(h, hg)
        return self.predict(h)
