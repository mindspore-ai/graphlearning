# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0(the "License");
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
"""gae"""
import mindspore as ms
from mindspore_gl import Graph
from mindspore_gl.nn import GNNCell
from mindspore_gl.nn import GCNConv

class GCNEncoder(GNNCell):
    """
    Graph convolution module, the number of convolution layers, convolution type
    and convolution parameters can be customized by the user.
    And the encoded features are obtained through forward propagation.
    For VGAE, the convolution of the last two layers will be executed in parallel to output the mean and variance.

    Args:
        data_feat_size: Data input feature dimension.
        hidden_dim: The output dimension of the convolution at each layer, shape :math:`(x, 2)`
        conv: Set convolution type.
        activate: The activation function of each layer of convolution, shape :math:`(x, 1)`
        name: Model name.

    Return:
        GNNCell: VGAE and GAE encoder models
    """
    def __init__(self,
                 data_feat_size: int,
                 hidden_dim_size: tuple = (32, 16),
                 conv: GNNCell = GCNConv,
                 activate: tuple = (ms.nn.ReLU(), None),
                 name: str = 'GAE'
                ):
        super().__init__()
        if not isinstance(data_feat_size, int):
            raise TypeError("The data_feat_size type is {},\
                            but it should be int.".format(type(data_feat_size)))
        if not isinstance(hidden_dim_size, tuple):
            raise TypeError("The hidden_dim_size type is {},\
                            but it should be tuple.".format(type(hidden_dim_size)))
        if not isinstance(activate, tuple):
            raise TypeError("The activate type is {},\
                            but it should be tuple.".format(type(activate)))
        if not isinstance(name, str):
            raise TypeError("The name type is {},\
                            but it should be str.".format(type(name)))
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

        Args:
            x(Tensor): The input node features, shape :math:`(node, feature_size)`
            in_deg(Tensor): In degree, shape :math:`(node)`
            out_deg(Tensor): Out degree, shape :math:`(node)`
            g(Graph): The input graph.

        Returns:
            "GAE":
                x: encoded features, shape :math:`(node, hidden_dim[-1])`
            "VGAE":
                x: encoded features, shape :math:`(node, hidden_dim[-1])`
                hidden_out: The result of the parallel execution of the last two layers,
                shape :math:`(2, node, hidden_dim[-1])`
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
        dropout_rate(float):Dropout ratio

    Return:
        GNNCell: VGAE and GAE decoder models
    """
    def __init__(self,
                 dropout_rate=0.0,
                 decoder_type='all'):
        super().__init__()
        if not isinstance(dropout_rate, float):
            raise TypeError("The dropout_rate type is {},\
                            but it should be float.".format(type(dropout_rate)))
        if not isinstance(decoder_type, str):
            raise TypeError("The decoder_type type is {},\
                            but it should be str.".format(type(decoder_type)))
        self.dropout = ms.nn.Dropout(p=dropout_rate)
        self.type = decoder_type

    def decoder_all(self, x, g: Graph):
        g = g
        x = self.dropout(x)
        adj_rec = ms.ops.matmul(x, x.transpose(1, 0))
        return adj_rec

    def decoder(self, x, index):
        x = self.dropout(x)

        adj_rec = x[index[0]] * x[index[1]]
        return adj_rec.sum(-1)

    def construct(self, x, index, g: Graph):
        """
        Construct function for InnerProductDecoder.


        Args:
            x(Tensor): x that needs an inner product operation, shape :math:`(node, feature_size)`
            g (Graph): The input graph.

        Returns:
            adj_rec(Tensor): object after inner product operation,shape :math:`(node, node)`
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
    And it has achieved very good results on datasets of classic citation networks(eg: cora, citeseer and pubmed).

    Use GCNEncoder and InnerProductDecoder modules to form a GAE model, and implement forward propagation of the network

    ..math::
        GCN(X, A)=\hat{A} ReLU(\hat{A} X W_{0})
        Z = GCN(X, A)
        A = \sigma(Z Z^{T})

    Args:
        encoder(Cell), Encoder Cell
        decoder(Cell), Decoder Cell

    Return:
        GNNCell: GAE model

    Example:
        >>> from gae import GCNEncoder, InnerProductDecoder, GAENet
        >>> encoder = GCNEncoder(4)
        >>> decoder = InnerProductDecoder()
        >>> net = GAENet(encoder, decoder)
    """

    def __init__(self,
                 encoder,
                 decoder):
        super().__init__()
        if not isinstance(encoder, GNNCell):
            raise TypeError("The encoder type is {},\
                            but it should be GNNCell.".format(type(encoder)))
        if not isinstance(decoder, GNNCell):
            raise TypeError("The decoder type is {},\
                            but it should be GNNCell.".format(type(decoder)))
        self.encoder = encoder
        self.decoder = decoder

    def construct(self, x, in_deg, out_deg, index, g: Graph):
        """
        Construct function for GAE.

        Args:
            x(Tensor): The input node features, shape :math:`(node, feature_size)`
            in_deg(Tensor): In degree, shape :math:`(node)`
            out_deg(Tensor): Out degree, shape :math:`(node)`
            g(Graph): The input graph.

        Returns:
            x(Tensor): Link prediction matrix, shape :math:`(node, node)`
        """
        if not isinstance(x, ms.Tensor):
            raise TypeError("The x data type is {},\
                            but it should be Tensor.".format(type(x)))
        if not isinstance(in_deg, ms.Tensor):
            raise TypeError("The in_deg data type is {},\
                            but it should be Tensor.".format(type(in_deg)))
        if not isinstance(out_deg, ms.Tensor):
            raise TypeError("The out_deg data type is {},\
                            but it should be Tensor.".format(type(out_deg)))
        x, _ = self.encoder(x, in_deg, out_deg, g)
        x = self.decoder(x, index, g)
        return x
