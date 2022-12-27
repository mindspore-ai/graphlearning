# Contents

- [Translating Embeddings](##translating-embeddings)
- [Datasets](##datasets)
- [Environment Requirements](##environment-requirements)
- [Quick Start](##hyperparameters)
- [Experiment results （Accuracy）](##experiment-results)

## Translating Embeddings(TransE)

Translating Embeddings is a knowledge embedding measure from paper
`Translating Embeddings for Modeling Multi-relational Data
<https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html>`.

This repository contains a MindSpore Graph Learning implementation of TransE.

## Datasets

This experiment is based on [FB15k datasets](https://www.microsoft.com/en-us/download/confirmation.aspx?id=52312).

## Environment Requirements

- MindSpore >= 1.6.0
- GraphLearning >= 0.1.0

## Quick Start

CUDA_VISIBLE_DEVICES=0 python model_zoo/transe/trainval.py --data_path {data_path}

## Hyperparameters


