# Contents

- Graph Attention Networks (GATv2)
- Datasets
- Environment Requirements
- Quick Start
- Experiment results (acc)

## Graph Attention Networks V2(GATv2)

Graph Attention Networks V2(GATv2) was proposed in 2021 by Brody, Shaked et al.
By introducing a simple fix by modifying the order of operations and proposing GATv2: a dynamic graph attention variant that is strictly more expressive than GAT. GATv2 outperforms GAT across 12
OGB and other benchmarks.

More detail about GAT can be found in:

[Brody S,  Alon U,  Yahav E. (2021). How Attentive are Graph Attention Networks? arXiv preprint arXiv:2105.14491.](https://arxiv.org/pdf/2105.14491.pdf)

This repository contains a implementation of GATv2 based on MindSpore and GraphLearning

## Datasets

The experiment is based on [Cora-ML](https://data.dgl.ai/dataset/cora_v2.zip), which was extracted in "Deep gaussian embedding of attributed graphs: Unsupervised inductive learning via ranking." ICLR 2018

## Environment Requirements

- MindSpore >= 1.6.0
- GraphLearning >= 0.1.0

## Quick Start

CUDA_VISIBLE_DEVICES=0 python model_zoo/gatv2/trainval_cora.py --data_path  {data_path}

## Experiment results

Cora dataset

Test acc: 0.8180