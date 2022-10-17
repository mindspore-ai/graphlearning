# Contents

- Laplace Graph Transformer Networks (LGTN)
- Datasets
- Environment Requirements
- Quick Start
- Experiment results

## Laplace Graph Transformer Networks (LGTN)

Graph Isomorphism Network(GIN) was proposed in 2020 and designed to  evaluate graph outlier detection: Peculiar observations and new insights. In order to better play the role of the GIN network, combine it with the transformer network and use Laplacian as an auxiliary encoder.

More detail about GIN and Transformer can be found in:

L. Zhao and L. Akoglu, “On using classification datasets to evaluate graph outlier detection: Peculiar observations and new insights,” arXiv preprint arXiv:2012.12931, 2020.

Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. Advances in Neural Information Processing Systems. 2017: 5998-6008.

This repository contains a implementation of LGTN based on MindSpore and GraphLearning.

## Datasets

The experiment is based on ogbg-molpcba, which is a molecular dataset sampled from PubChem BioAssay. It is a graph prediction dataset from the Open Graph Benchmark (OGB).

## Environment Requirements

- MindSpore >= 1.6.0
- GraphLearning >= 0.1.0

## Quick Start

CUDA_VISIBLE_DEVICES=0 python recommendation/lp_graphtrans/train_lgtn.py

## Experiment results

ogbg-molpcba dataset