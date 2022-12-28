# Contents

- [Heterogeneours Graph Transformer](##heterogeneous-graph-transformer)
- [Datasets](##datasets)
- [Environment Requirements](##environment-requirements)
- [Quick Start](##hyperparameters)
- [Experiment results （Accuracy）](##experiment-results)

## Heterogeneous Graph Transformer

Heterogeneours Graph Transformer is a graph neural network architecture that can deal with large-scale heterogeneous and dynamic graphs.
More detail about this model can be found in:

Ziniu Hu, Yuxiao Dong, Kuansan Wang, Yizhou Sun, "Heterogeneous Graph Transformer" (WWW'20)

This repository contains a MindSpore Graph Learning implementation of HGT based upon the author's original PyG implementation (https://github.com/acbull/pyHGT).

## Datasets

This experiment is based on [ACM datasets](https://data.dgl.ai/dataset/ACM.mat). As the ACM datasets doesn't have input feature, we simply randomly assign features for each node. Such process can be simply replaced by any prepared features.

## Environment Requirements

- MindSpore >= 1.6.0
- GraphLearning >= 0.1.0

## Quick Start

CUDA_VISIBLE_DEVICES=0 python model_zoo/hgt/trainval.py --data_path  {data_path}

## Hyperparameters

## Experiment results

| Model        | Test Accuracy    | # Parameter  |
| ---------    | ---------------  | -------------|
| 2-layer HGT  | 0.427 ± 0.01   |  2,176,324   |
