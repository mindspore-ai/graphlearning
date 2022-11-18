# Contents

- Neural Message Passing for Quantum Chemistry (MPNN)
- Datasets
- Environment Requirements
- Quick Start
- Experiment results (mae)

## Neural Message Passing for Quantum Chemistry (MPNN)

More detail about Neural Message Passing for Quantum Chemistry (MPNN) can be found in:

[Justin G, Samuel S. S, Patrick F. R, et al. Neural Message Passing for Quantum Chemistry](https://arxiv.org/pdf/1704.01212.pdf)

This repository contains a implementation of MPNN based on MindSpore and GraphLearning

## Datasets

This experiment is based on chemistry dataset Alchemy ([dev](https://alchemy.tencent.com/data/dev_v20190730.zip)/[valid](https://alchemy.tencent.com/data/valid_v20190730.zip)).

## Environment Requirements

- MindSpore >= 1.6.0
- GraphLearning >= 0.1.0

## Quick Start

CUDA_VISIBLE_DEVICES=0 python model_zoo/mpnn/trainval.py --data_path  {data_path}

## Experiment results

Alchemy dataset

Val mae: 0.15