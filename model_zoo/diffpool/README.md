# Contents

- DIFFPOOL
- Datasets
- Environment Requirements
- Quick Start
- Experiment results (acc)

## DIFFPOOL

DiffPool, a differentiable graph pooling module that can generate hierarchical representations of graphs and can be combined with various graph neural network architectures in an end-to-end fashion. DiffPool learns a differentiable soft cluster assignment for nodes at each layer of a deep GNN, mapping nodes to a set of clusters, which then form the coarsened input for the next GNN layer.

More detail about DIFFPOOL can be found in:

[Ying Z, You J, Morris C, et al. Hierarchical graph representation learning with differentiable pooling[J]. Advances in neural information processing systems, 2018, 31.](https://proceedings.neurips.cc/paper/2018/file/e77dbaf6759253c7c6d0efc5690369c7-Paper.pdf)

This repository contains a implementation of DIFFPOOL based on MindSpore and GraphLearning

## Datasets

The experiment is based on [ENZYMES](https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/ENZYMES.zip).

## Environment Requirements

- MindSpore >= 1.6.0
- GraphLearning >= 0.1.0

## Quick Start

CUDA_VISIBLE_DEVICES=0 python model_zoo/diffpool/trainval_enzymes.py --data_path  {data_path}

## Experiment results

ENZEMES dataset

Best val acc: 0.733