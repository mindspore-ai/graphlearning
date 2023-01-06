# Contents

- Graph Auto-Encoders (GAE)
- Datasets
- Environment Requirements
- Quick Start
- Experiment results (AUC、AP)

## Graph Auto-Encoders (GAE)

Graph Auto-Encoders (GCN) was proposed in 2016 and designed to do semi-supervised learning on graph-structured data. The model utilizes latent variables and is able to learn interpretable latent representations of undirected graphs. And gae is the non-probabilistic graph auto-encoder (GAE) model.

This repository contains a implementation of GAE based on MindSpore and GraphLearning

- Paper link：https://arxiv.org/abs/1611.07308
- Author's code repo：https://github.com/tkipf/gae

## Datasets

The official description of the dataset used in this experiment is as follows:

**Cora**: The Cora dataset consists of 2708 scientific publications classified into one of seven classes. The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words. The README file in the dataset provides more details.

**Citeseer**: The CiteSeer dataset consists of 3312 scientific publications classified into one of six classes. The citation network consists of 4732 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 3703 unique words. The README file in the dataset provides more details.

**PubMed**: The Pubmed Diabetes dataset consists of 19717 scientific publications from PubMed database pertaining to diabetes classified into one of three classes. The citation network consists of 44338 links. Each publication in the dataset is described by a TF/IDF weighted word vector from a dictionary which consists of 500 unique words. The README file in the dataset provides more details.

- Official dataset address: https://linqs.org/datasets/
- Address after data processing: [cora_v2](https://data.dgl.ai/dataset/cora_v2.zip), [citeseer](https://data.dgl.ai/dataset/citeseer.zip), [pubmed](https://data.dgl.ai/dataset/pubmed.zip)

## Environment Requirements

- MindSpore >= 1.7.0
- GraphLearning >= 0.1.0

## Quick Start

Run with following (available data_name: "Cora", "Citeseer", "PubMed")

CUDA_VISIBLE_DEVICES=0 python model_zoo/gae/trainval.py --epochs=200 --data_name 'Cora' --lr=0.01 --weight_decay=0.0 --dropout=1.0 --hidden1_dim=32 --hidden2_dim 16 --mode1 "undirected" --mode2 "dense" --data_path {data_path}

## Experiment results

Use *area under the ROC curve* (AUC) and *average precision* (AP) scores for each model on the test set. Numbers show mean results and standard error for 10 runs with random initializations on fixed dataset splits.

### Result in dgl

| Dataset  | AUC            | AP            |
| -------- | -------------- | ------------- |
| Cora     | 91.28 $\pm$ 0.01 | 92.54 $\pm$ 0.01 |
| Citeseer | 89.36 $\pm$ 0.01 | 89.98 $\pm$ 0.01 |
| Pubmed   | 96.17 $\pm$ 0.01 | 96.29 $\pm$ 0.01 |

### Result in mindspore-gl(imitate dgl for training)

| Dataset  | AUC            | AP            |
| -------- | -------------- | ------------- |
| Cora     | 91.24 $\pm$ 0.01 | 91.28 $\pm$ 0.01 |
| Citeseer | 90.25 $\pm$ 0.01 | 90.40 $\pm$ 0.01 |
| Pubmed   | 96.03 $\pm$ 0.01 | 96.03 $\pm$ 0.01 |

### Result in pytorch-geometric

| Dataset  | AUC            | AP             |
| -------- | -------------- | -------------- |
| Cora     | 90.94 $\pm$ 0.01 | 91.08 $\pm$ 0.01  |
| Citeseer | 88.16 $\pm$ 0.01 | 88.70 $\pm$ 0.01  |
| Pubmed   | 96.29 $\pm$ 0.01 | 96.38 $\pm$ 0.01  |

### Result in mindspore-gl(imitate pyg for training)

| Dataset  | AUC            | AP             |
| -------- | -------------- | --------------        |
| Cora     | 91.08 $\pm$ 0.01 | 92.06 $\pm$ 0.01  |
| Citeseer | 88.92 $\pm$ 0.01 | 89.29 $\pm$ 0.01  |
| Pubmed   | 96.04 $\pm$ 0.01 | 95.01 $\pm$ 0.01  |
