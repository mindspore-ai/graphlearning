# Contents

- Heterogeneours Graph Transformer
- Datasets
- Environment Requirements
- Quick Start
- Experiment results (f1 score)

## Deepwalk

DeepWalk is a approach for learning latent representations of vertices in a network. These latent representations encode social relations in a continuous vector space, which is easily exploited by statistical models. DeepWalk generalizes recent advancements in language modeling and unsupervised feature learning (or deep learning) from sequences of words to graphs.

More detail about this model can be found in:

Perozzi B, Al-Rfou R, Skiena S, "Deepwalk: Online learning of social representations" (SIGKDD 2014)

This repository contains a implementation of Deepwalk based on MindSpore and GraphLearning

## Datasets

This experiment is based on BlogCatalog datasets (https://dl.acm.org/doi/abs/10.1145/1557019.1557109). BlogCatalog is a network of social relationships provided by blogger authors. The labels represent the topic categories provided by the authors.

## Environment Requirements

- MindSpore >= 1.6.0
- GraphLearning >= 0.1.0

## Quick Start

CUDA_VISIBLE_DEVICES=0 python model_zoo/deepwalk/train_blog_catalog.py --data_path  {data_path}

CUDA_VISIBLE_DEVICES=0 python model_zoo/deepwalk/eval_blog_catalog.py --data_path  {data_path}

## Experiment results

Test: micro f1: 0.3818, macro f1: 0.2560