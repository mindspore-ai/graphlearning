# Contents

- STGCN
- Datasets
- Environment Requirements
- Quick Start
- Experiment results (mse)

## STGCN

Spatio-Temporal Graph Convolutional Networks (STGCN) can tackle the time series prediction problem in traffic domain. Experiments show that STGCN effectively captures comprehensive spatio-temporal correlations through modeling multi-scale traffic networks.

More detail about STGCN can be found in:

[Yu B, Yin H, Zhu Z. Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting[J]. arXiv preprint arXiv:1709.04875, 2017.](https://arxiv.org/pdf/1709.04875.pdf)

This repository contains a implementation of STGCN based on MindSpore and GraphLearning

## Datasets

This experiment is based on [metr-la](https://graphmining.ai/temporal_datasets/METR-LA.zip)

## Environment Requirements

- MindSpore >= 1.6.0
- GraphLearning >= 0.1.0

## Quick Start

GPU:
CUDA_VISIBLE_DEVICES=0 python model_zoo/stgcn/trainval_metr.py --data_path  {data_path}

Ascend:
python model_zoo/stgcn/trainval_metr.py --data_path  {data_path} --device Ascend

## Experiment results

Metr_LA dataset

Test MSE: 0.4