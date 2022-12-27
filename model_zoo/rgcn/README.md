# Contents

- [Heterogeneours Graph Transformer](##relation-gcn)
- [Datasets](##datasets)
- [Environment Requirements](##environment-requirements)
- [Quick Start](##hyperparameters)
- [Experiment results （Accuracy）](##experiment-results)

## Relational-GCN

This repository contains a MindSpore Graph Learning implementation of [RGCN](https://arxiv.org/abs/1703.06103) for node classification . The authors' implementation can be found [here](https://github.com/tkipf/relational-gcn).

## Datasets

This experiment is based on ACM.  As the ACM datasets don't have input feature, we simply randomly assign features for each node. Such process can be simply replaced by any prepared features.

## Environment Requirements

- MindSpore >= 1.6.0
- GraphLearning >= 0.1.0

## Quick Start

CUDA_VISIBLE_DEVICES=0 python model_zoo/rgcn/trainval.py --data_path  {data_path}

## Hyperparameters

## Experiment results

We train the RGCN model for 100 epochs and report the best accuracy on the test dataset.

| Model        | Test Accuracy    |
| ---------    | ---------------  |
| 2-layer rgcn | 0.385   |
