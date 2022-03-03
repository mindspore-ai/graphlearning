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
"""Predict training"""
import argparse
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.context as context

from mindspore_gl.dataset import BlogCatalog
from mindspore_gl.dataloader import RandomBatchSampler, Dataset, DataLoader
from sklearn.metrics import f1_score


class Model(nn.Cell):
    """Model"""
    def __init__(self, embed_size, num_classes):
        super(Model, self).__init__()
        self.dense = nn.Dense(embed_size, num_classes)

    def construct(self, node_emb):
        logits = self.dense(node_emb)
        return logits


class LossNet(nn.Cell):
    """Lossnet"""
    def __init__(self, net):
        super(LossNet, self).__init__()
        self.net = net
        self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)

    def construct(self, node_emb, label):
        logits = self.net(node_emb)
        loss = self.loss_fn(logits, label)
        loss = ops.ReduceMean()(loss)
        return loss


class PredictDataset(Dataset):
    """Predict dataset"""
    def __init__(self, nodes, label):
        self.data = nodes
        self.label = label
        self.datalen = len(nodes)

    def __getitem__(self, batch_idxs):
        return np.take(self.data, batch_idxs, 0), np.take(self.label, batch_idxs, 0)

    def __len__(self):
        return self.datalen


def main(arguments):
    if arguments.fuse:
        context.set_context(device_target="GPU", save_graphs=True, save_graphs_path="./computational_graph/",
                            mode=context.GRAPH_MODE, enable_graph_kernel=True)
    else:
        context.set_context(device_target="GPU")

    dataset = BlogCatalog(arguments.data_path)

    with np.load(arguments.data_path + arguments.save_file_name) as npz_file:
        embedding = npz_file['embedding_weight']
        embedding_table = ms.Parameter(ms.Tensor(embedding, ms.float32), requires_grad=False)
    emb = nn.Embedding(
        ops.Shape()(embedding_table)[0],
        arguments.embed_size,
        embedding_table=embedding_table)

    vocab = dataset.vocab
    label = dataset.node_label

    node_feat = emb(ms.Tensor(vocab, ms.int32)).asnumpy()

    vocab_size = len(label)
    mask = np.random.permutation(vocab_size)
    train_mask, test_mask = mask[:int(vocab_size * arguments.train_ratio)], \
                            mask[int(vocab_size * arguments.train_ratio):]

    predict_dataset = PredictDataset(node_feat, label)

    train_sampler = RandomBatchSampler(
        data_source=train_mask,
        batch_size=arguments.batch_size)

    train_dataloader = DataLoader(
        dataset=predict_dataset,
        sampler=train_sampler,
        num_workers=0,
        persistent_workers=False)

    test_sampler = RandomBatchSampler(
        data_source=test_mask,
        batch_size=arguments.batch_size)

    test_dataloader = DataLoader(
        dataset=predict_dataset,
        sampler=test_sampler,
        num_workers=0,
        persistent_workers=False)

    net = Model(arguments.embed_size, dataset.num_classes)
    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=arguments.multiclass_learning_rate)
    loss = LossNet(net)
    train_net = nn.TrainOneStepCell(loss, optimizer)

    for e in range(arguments.epoch):
        train_net.set_train(True)
        train_loss = 0
        for data in train_dataloader:
            node_feat, label = data
            node_feat, label = ms.Tensor.from_numpy(node_feat), ms.Tensor.from_numpy(label)
            train_loss += train_net(node_feat, label).asnumpy()
        train_loss /= len(train_dataloader)

        train_net.set_train(False)
        test_pred, test_label = [], []
        for data in test_dataloader:
            node_feat, label = data
            node_feat = ms.Tensor.from_numpy(node_feat)
            logits = net(node_feat).asnumpy()
            pred = np.argmax(logits, axis=1)
            test_label.extend(label.tolist())
            test_pred.extend(pred.tolist())
        test_micro_f1 = f1_score(test_label, test_pred, average='micro')
        test_macro_f1 = f1_score(test_label, test_pred, average='macro')

        print("Epoch {}, train loss: {}".format(e+1, train_loss))
        print("Test: micro f1: {:.4f}, macro f1: {:.4f}".format(test_micro_f1, test_macro_f1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deepwalk')
    parser.add_argument("--data_path", type=str, help="path to dataloader")
    parser.add_argument("--save_file_name", type=str, default="deepwalk_embedding.npz",
                        help="path to save embedding weight")
    parser.add_argument("--dataset", type=str, default="BlogCatalog", help="dataset")
    parser.add_argument("--gpu", type=int, default=1, help="which gpu to use")
    parser.add_argument("--epoch", type=int, default=40, help="number of epoch")
    parser.add_argument("--embed_size", type=int, default=128, help="size of embedding")
    parser.add_argument("--batch_size", type=int, default=32, help="number of batch size")
    parser.add_argument("--multiclass_learning_rate", type=float, default=0.001, help="multiclass learning rate")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="ratio of train data")
    parser.add_argument('--fuse', type=bool, default=False, help="whether to use graph mode")
    args = parser.parse_args()
    print(args)
    main(args)
