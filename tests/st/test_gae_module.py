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
""" test model """
import numpy as np
import mindspore as ms
from mindspore_gl.nn import GCNConv
from mindspore_gl import GraphField
from mindspore_gl import Graph
from mindspore_gl.nn import GNNCell
import pytest

class GCNEncoder(GNNCell):
    """
    Graph convolution module, the number of convolution layers,
    convolution type and convolution parameters can be customized by the user.
    And the encoded features are obtained through forward propagation.
    For VGAE, the convolution of the last two layers will be executed in parallel to output the mean and variance.

    Args:
        data_feat_size: Data input feature dimension.
        hidden_dim: The output dimension of the convolution at each layer, shape: (x, 2)
        conv：Set convolution type.
        activate：The activation function of each layer of convolution, shape: (x, 1)
        name：Model name.
    """
    def __init__(self,
                 data_feat_size: int,
                 hidden_dim_size: tuple = (32, 16),
                 conv: GNNCell = GCNConv,
                 activate: tuple = (ms.nn.ReLU(), None),
                 name: str = 'GAE'
                ):
        super().__init__()
        self.name = name
        layer = []
        if name == 'GAE':
            for i in range(len(activate)):
                if i == 0:
                    layer.append(conv(data_feat_size, hidden_dim_size[i], activate[i]))
                else:
                    layer.append(conv(hidden_dim_size[i-1], hidden_dim_size[i], activate[i]))
        elif name == 'VGAE':
            for i in range(len(activate)):
                if i == 0:
                    layer.append(conv(data_feat_size, hidden_dim_size[i], activate[i]))
                elif i < len(activate) - 1:
                    layer.append(conv(hidden_dim_size[i-1], hidden_dim_size[i], activate[i]))
                else:
                    layer.append(conv(hidden_dim_size[i-2], hidden_dim_size[i], activate[i]))
        self.layer = ms.nn.SequentialCell(layer)

    def construct(self, x, in_deg, out_deg, g: Graph):
        """
        Construct function for GCNEncoder.

        Args：
            x(Tensor): The input node features.,shape: (node, feature_size)
            in_deg(Tensor): In degree, shape: (node)
            out_deg(Tensor): Out degree, shape: (node)
            g (Graph): The input graph.

        Returns：
            "GAE":
                x: encoded features, shape:(node, hidden_dim[-1])
            "VGAE":
                x: encoded features，shape:(node, hidden_dim[-1])
                hidden_out：The result of the parallel execution of the last two layers, shape:(2, node, hidden_dim[-1])
        """
        hidden_out = []
        if self.name == 'GAE':
            for cell in self.layer.cell_list:
                x = cell(x, in_deg, out_deg, g)
        elif self.name == 'VGAE':
            for cell in self.layer.cell_list[:-2]:
                x = cell(x, in_deg, out_deg, g)
            hidden_out.append(self.layer.cell_list[-2](x, in_deg, out_deg, g))
            hidden_out.append(self.layer.cell_list[-1](x, in_deg, out_deg, g))

        return x, hidden_out

class InnerProductDecoder(GNNCell):
    """
    Inner product encoder module, which performs inner product operation on the input object
    and returns the object after performing the inner product.

    Args:
        dropout_rate(float):Drop ratio
    """
    def __init__(self,
                 dropout_rate=0.0,
                 decoder_type='all'):
        super().__init__()
        self.dropout = ms.nn.Dropout(p=dropout_rate)
        self.type = decoder_type

    def decoder_all(self, x, g: Graph):
        g = g
        x = self.dropout(x)
        transpose = ms.ops.Transpose()
        adj_rec = ms.ops.matmul(x, transpose(x, (1, 0)))
        return adj_rec

    def decoder(self, x, index):
        x = self.dropout(x)

        adj_rec = x[index[0]] * x[index[1]]
        return adj_rec.sum(-1)

    def construct(self, x, index, g: Graph):
        """
        Construct function for InnerProductDecoder.


        Args:
            x(Tensor): x that needs an inner product operation, shape:(node, feature_size)
            g (Graph): The input graph.

        Returns:
            adj_rec(Tensor): object after inner product operation,shape:(node, node)
        """
        if self.type == 'all':
            adj_rec = self.decoder_all(x, g.src_idx)
        else:
            adj_rec = self.decoder(x, index)
        return adj_rec

class GAENet(GNNCell):
    r"""
    GAE is a classic unsupervised model in graph neural networks,
    which mainly utilizes the coding features obtained by graph convolutional encoders.
    The inner product decoder is used to reconstruct the samples to learn the latent representation of the graph.
    And it has achieved very good results on datasets of classic citation networks (eg: cora, citeseer and pubmed).

    Use GCNEncoder and InnerProductDecoder modules to form a GAE model, and implement forward propagation of the network

    ..math::
        GCN(X, A)=\hat{A} ReLU(\hat{A} X W_{0})
        Z = GCN(X, A)
        A = \sigma (Z Z^{T})

    Args:
        encoder(Cell), Encoder Cell
        decoder(Cell), Decoder Cell
    """

    def __init__(self,
                 encoder,
                 decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def construct(self, x, in_deg, out_deg, index, g: Graph):
        """
        Construct function for GAE.

        Args：
            x(Tensor): The input node features.,shape: (node, feature_size)
            in_deg(Tensor): In degree, shape: (node)
            out_deg(Tensor): Out degree, shape: (node)
            g (Graph): The input graph.

        Returns:
            x(Tensor): Link prediction matrix, shape:(node, node)
        """

        x, _ = self.encoder(x, in_deg, out_deg, g)
        x = self.decoder(x, index, g)
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_encoder():
    """
    Feature:Test whether the input and output dimensions of the encoder are correct.

    Description:
    initialize edge
    node = 7
    edge = 6
    src = [0, 1, 2, 3, 4, 5, 6]
    dst = [7, 6, 5, 4, 3, 2, 1]
    indeg = [0, 1, 1, 1, 1, 1, 1]
    outdeg = [1, 1, 1, 1, 1, 1, 0]

    Expectation:
    output.shape == expected.shape
    """
    node_feat = ms.Tensor([
        # graph 1:
        [1, 2, 3, 4],
        [2, 4, 1, 3],
        [1, 3, 2, 4],
        [9, 7, 5, 8],
        [8, 7, 6, 5],
        [8, 6, 4, 6],
        [1, 2, 1, 1],
    ], ms.float32)

    n_nodes = 7
    n_edges = 6
    src_idx = ms.Tensor([0, 1, 2, 3, 4, 5, 6], ms.int32)
    dst_idx = ms.Tensor([1, 2, 3, 4, 5, 6, 0], ms.int32)
    in_deg = ms.Tensor([0, 1, 1, 1, 1, 1, 1], ms.int32)
    out_deg = ms.Tensor([1, 1, 1, 1, 1, 1, 0], ms.int32)
    g = GraphField(ms.Tensor(src_idx, dtype=ms.int32), ms.Tensor(dst_idx, dtype=ms.int32),
                   int(n_nodes), int(n_edges))

    encoder = GCNEncoder(4)
    ret, _ = encoder(node_feat, in_deg, out_deg, *g.get_graph())

    expected = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]
    assert np.array(ret).shape == np.array(expected).shape

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_decoder():
    """
    Feature:Test whether the input and output of the decoder model module are correct.

    Description:
    initialize edge

    node = 7
    edge = 6
    src = [0, 1, 2, 3, 4, 5, 6]
    dst = [1, 2, 3, 4, 5, 6, 7]

    Expectation:
    output.shape == expected.shape
    """
    x = ms.Tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ], ms.float32)

    n_nodes = 7
    n_edges = 6
    src_idx = ms.Tensor([0, 1, 2, 3, 4, 5, 6], ms.int32)
    dst_idx = ms.Tensor([1, 2, 3, 4, 5, 6, 3], ms.int32)
    index = 0
    g = GraphField(ms.Tensor(src_idx, dtype=ms.int32), ms.Tensor(dst_idx, dtype=ms.int32),
                   int(n_nodes), int(n_edges))


    decoder = InnerProductDecoder()
    ret = decoder(x, index, *g.get_graph()).asnumpy().tolist()

    expected = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ]
    assert np.array(ret).shape == np.array(expected).shape

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gae_model():
    """
    Feature:Test whether the input and output of the gae model module are correct.

    Description:
    initialize edge
    node = 7
    edge = 6
    src = [0, 1, 2, 3, 4, 5, 6]
    dst = [1, 2, 3, 4, 5, 6, 7]
    indeg = [0, 1, 1, 1, 1, 1, 1]
    outdeg = [1, 1, 1, 1, 1, 1, 0]

    Expectation:
    output.shape == expected.shape
    """
    node_feat = ms.Tensor([
        # graph 1:
        [1, 2, 3, 4],
        [2, 4, 1, 3],
        [1, 3, 2, 4],
        [9, 7, 5, 8],
        [8, 7, 6, 5],
        [8, 6, 4, 6],
        [1, 2, 1, 1],
    ], ms.float32)

    n_nodes = 7
    n_edges = 6
    index = 0
    src_idx = ms.Tensor([0, 1, 2, 3, 4, 5, 6], ms.int32)
    dst_idx = ms.Tensor([1, 2, 3, 4, 5, 6, 2], ms.int32)
    in_deg = ms.Tensor([0, 1, 1, 1, 1, 1, 1], ms.int32)
    out_deg = ms.Tensor([1, 1, 1, 1, 1, 1, 0], ms.int32)
    g = GraphField(ms.Tensor(src_idx, dtype=ms.int32), ms.Tensor(dst_idx, dtype=ms.int32),
                   int(n_nodes), int(n_edges))

    encoder = GCNEncoder(4)
    decoder = InnerProductDecoder()
    ret = GAENet(encoder, decoder)(node_feat, in_deg, out_deg, index, *g.get_graph())

    expected = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ]
    assert np.array(ret).shape == np.array(expected).shape
