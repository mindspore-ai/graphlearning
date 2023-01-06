# Contents

- Graph Convolutional Networks (GCN)
- Datasets
- Environment Requirements
- Quick Start
- Experiment results

## Graph Convolutional Networks (GCN)

Graph Convolutional Networks (GCN) was proposed in 2016 and designed to do semi-supervised learning on graph-structured data. A scalable approach based on an efficient variant of convolutional neural networks which operate directly on graphs was presented. The model scales linearly in the number of graph edges and learns hidden layer representations that encode both local graph structure and features of nodes.

More detail about GCN can be found in:

[Thomas N. Kipf, Max Welling. 2016. Semi-Supervised Classification with Graph Convolutional Networks.](https://arxiv.org/pdf/1609.02907.pdf) In ICLR 2016.

This repository contains a implementation of GCN based on MindSpore and GraphLearning

## Datasets

The experiment is based on [Cora-ML](https://data.dgl.ai/dataset/cora_v2.zip), which was extracted in "Deep gaussian embedding of attributed graphs: Unsupervised inductive learning via ranking." ICLR 2018.

For distributed training, the experiment is based on [Reddit dataset](https://data.dgl.ai/dataset/reddit.zip).

## Environment Requirements

- MindSpore >= 1.6.0
- GraphLearning >= 0.1.0

## Quick Start

`CUDA_VISIBLE_DEVICES=0 python model_zoo/gcn/trainval_cora.py --data_path  {data_path}`

## Distributed Training

GPU:\
`bash model_zoo/gcn/distributed_run.sh GPU {DATA_PATH}`

Ascend:\
`bash model_zoo/gcn/distributed_run.sh Ascend {DATA_PATH}`

## Experiment results

**Cora dataset**

Test acc: 0.8280

**Reddit dataset**

The results on Reddit dataset are shown in the table following.

- NPU

  | device    |   1*Ascend 910     | 4*Ascend 910 |
  | --------- | :----------------: | :----------: |
  | acc       |       0.9268       |      0.9247  |
  | used time |       1064s        |     280s     |

- GPU

  | device | 1*V100 | 4*V100 |
  | --------- | :----------------: |  :----------: |
  | acc       |       0.9224       | 0.9249 |
  | used time |       80s       |  55s |