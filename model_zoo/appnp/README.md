# Contents

- APPNP
- Datasets
- Environment Requirements
- Quick Start
- Experiment results (acc)

## APPNP

Personalized propagation of neural predictions (PPNP), and its fast approximation (APPNP), use the relationship between graph convolutional networks (GCN) and PageRank to derive an improved propagation scheme based on personalized PageRank. It leverages a large, adjustable neighborhood for classification and can be easily combined with any neural network.

More detail about APPNP can be found in:

[Klicpera J , Bojchevski A, S Günnemann. Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://arxiv.org/pdf/1810.05997.pdf)

This repository contains a implementation of APPNP based on MindSpore and GraphLearning

## Datasets

The experiment is based on [Cora-ML](https://data.dgl.ai/dataset/cora_v2.zip), which was extracted in "Deep gaussian embedding of attributed graphs: Unsupervised inductive learning via ranking." ICLR 2018

## Environment Requirements

- MindSpore >= 1.6.0
- GraphLearning >= 0.1.0

## Quick Start

CUDA_VISIBLE_DEVICES=0 python model_zoo/appnp/trainval_cora.py --data_path  {data_path}

## Experiment results

Cora dataset

Test acc: 0.8350