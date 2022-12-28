# Contents

- [Heterogeneours Graph Attention Network](##heterogeneous-graph-attention-network)
- [Datasets](##datasets)
- [Environment Requirements](##environment-requirements)
- [Quick Start](##hyperparameters)
- [Experiment results （Accuracy）](##experiment-results)

## Heterogeneous Graph Attention Network

This repository contains a MindSpore Graph Learning implementation of [Heterogeneous Graph Attention Network(HAN)](https://dl.acm.org/doi/10.1145/3308558.3313562). The authors' implementation can be found [here](https://github.com/Jhy1993/HAN).

## Datasets

This experiment is based on ACM_3025. The dataset can be downloaded from the [author's repository](https://github.com/Jhy1993/HAN).

## Environment Requirements

- MindSpore >= 1.6.0
- GraphLearning >= 0.1.0

## Quick Start

CUDA_VISIBLE_DEVICES=0 python model_zoo/han/trainval_acm.py --data_path  {data_path}

## Hyperparameters

## Experiment results

We train the HAN model for 100 epochs and report the best accuracy on the test dataset.

| Model        | Test Accuracy    |
| ---------    | ---------------  |
| 8-layer HAN  | 0.875   |
