# Contents

- Graph Isomorphism Network (GIN)
- Datasets
- Environment Requirements
- Quick Start
- Experiment results (acc)

## Graph Isomorphism Network (GIN)

More detail about Graph Isomorphism Network (GIN) can be found in:

[Xu K, Hu W, Leskovec J, et al. How powerful are graph neural networks?](https://arxiv.org/pdf/1810.00826.pdf)

This repository contains a implementation of GIN based on MindSpore and GraphLearning

## Datasets

This experiment is based on social network dataset ([IMDB-BINARY](https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/IMDB-BINARY.zip)).

## Environment Requirements

- MindSpore >= 1.6.0
- GraphLearning >= 0.1.0

## Quick Start

CUDA_VISIBLE_DEVICES=0 python model_zoo/gin/trainval_imdb_binary.py --data_path  {data_path}

## Experiment results

IMDB-BINARY dataset

Test acc: 0.72```