# Contents

- GeniePath
- Datasets
- Environment Requirements
- Quick Start
- Experiment results (acc)

## GeniePath

GeniePath is a scalable approach for learning adaptive receptive fields of neural networks defined on permutation invariant graph data. GeniePath propose an adaptive path layer consists of two complementary functions designed for breadth and depth exploration respectively, where the former learns the importance of different sized neighborhoods, while the latter extracts and filters signals aggregated from neighbors of different hops away.

More detail about this model can be found in:

Liu Z, Chen C, Li L, et al. Geniepath: Graph neural networks with adaptive receptive paths (AAAI 2019).

This repository contains a implementation of Deepwalk based on MindSpore and GraphLearning

## Datasets

This experiment is based on node classification datasets (Pubmed citation network dataset and Protein-Protein Interaction dataset).

## Environment Requirements

- MindSpore >= 1.6.0
- GraphLearning >= 0.1.0

## Quick Start

CUDA_VISIBLE_DEVICES=0 python model_zoo/geniepath/trainval_pubmed.py --data_path  {data_path}

CUDA_VISIBLE_DEVICES=0 python model_zoo/geniepath/trainval_ppi.py --data_path  {data_path}

## Experiment results

Pubmed citation network dataset

Test acc: 0.6490