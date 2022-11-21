# Contents

- GraphSAGE
- Datasets
- Environment Requirements
- Quick Start
- Experiment results (acc)

## GraphSAGE

GraphSAGE is a general inductive framework that leverages node feature information (e.g., text attributes) to efficiently generate node embeddings for previously unseen data. Instead of training individual embeddings for each node, GraphSAGE learns a function that generates embeddings by sampling and aggregating features from a node's local neighborhood.

More detail about GraphSAGE can be found in:

[Hamilton W L, Ying R, Leskovec J. Inductive representation learning on large graphs (Neural Information Processing Systems, 2017)](https://proceedings.neurips.cc/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf)

This repository contains a implementation of GraphSAGE based on MindSpore and GraphLearning

## Datasets

This experiment is based on [Reddit dataset](https://data.dgl.ai/dataset/reddit.zip)

## Environment Requirements

- MindSpore >= 1.6.0
- GraphLearning >= 0.1.0

## Quick Start

GPU:\
CUDA_VISIBLE_DEVICES=0 python model_zoo/graphsage/trainval_reddit.py --data_path  {data_path}

Ascend:\
python model_zoo/graphsage/trainval_reddit.py --data_path  {data_path} --device Ascend

## Distributed Training

GPU:\
bash distributed_run.sh GPU {DATA_PATH}

Ascend:\
bash distributed_run.sh Ascend {DATA_PATH} {RANK_START} {RANK_SIZE} {RANK_TABLE_FILE}

## Experiment results

Reddit dataset

Test acc: 0.93