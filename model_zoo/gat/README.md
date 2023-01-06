# Contents

- Graph Attention Networks (GAT)
- Datasets
- Environment Requirements
- Quick Start
- Experiment results (acc)

## Graph Attention Networks (GAT)

Graph Attention Networks(GAT) was proposed in 2017 by Petar Veličković et al. By leveraging masked self-attentional layers to address shortcomings of prior graph based method, GAT achieved or matched state of the art performance on both transductive datasets like Cora and inductive dataset like PPI.

More detail about GAT can be found in:

[Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2017). Graph attention networks. arXiv preprint arXiv:1710.10903.](https://arxiv.org/pdf/1710.10903.pdf)

This repository contains a implementation of GAT based on MindSpore and GraphLearning

## Datasets

The experiment is based on [Cora-ML](https://data.dgl.ai/dataset/cora_v2.zip), which was extracted in "Deep gaussian embedding of attributed graphs: Unsupervised inductive learning via ranking." ICLR 2018

## Environment Requirements

- MindSpore >= 1.6.0
- GraphLearning >= 0.1.0

## Quick Start

CUDA_VISIBLE_DEVICES=0 python model_zoo/gat/trainval_cora.py --data_path  {data_path}

## Experiment results

Cora dataset

Test acc: 0.8180