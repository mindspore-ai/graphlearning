# Contents

- Graph Convolutional Networks (GCN)
- Datasets
- Environment Requirements
- Quick Start
- Experiment results (acc)

## Graph Convolutional Networks (GCN)

Graph Convolutional Networks (GCN) was proposed in 2016 and designed to do semi-supervised learning on graph-structured data. A scalable approach based on an efficient variant of convolutional neural networks which operate directly on graphs was presented. The model scales linearly in the number of graph edges and learns hidden layer representations that encode both local graph structure and features of nodes.

More detail about GCN can be found in:

Thomas N. Kipf, Max Welling. 2016. Semi-Supervised Classification with Graph Convolutional Networks. In ICLR 2016.

This repository contains a implementation of GCN based on MindSpore and GraphLearning

## Datasets

The experiment is based on Cora-ML, which was extracted in "Deep gaussian embedding of attributed graphs: Unsupervised inductive learning via ranking." ICLR 2018

## Environment Requirements

- MindSpore >= 1.6.0
- GraphLearning >= 0.1.0

## Quick Start

CUDA_VISIBLE_DEVICES=0 python model_zoo/gcn/trainval_cora.py --data_path  {data_path}

## Experiment results

Cora dataset

Test acc: 0.8280