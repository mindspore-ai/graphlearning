# Contents

- Model
- Datasets
- Environment Requirements
- Quick Start
- Experiment results (acc)

## Model

- APPNP

Personalized propagation of neural predictions (PPNP), and its fast approximation (APPNP), use the relationship between graph convolutional networks (GCN) and PageRank to derive an improved propagation scheme based on personalized PageRank. It leverages a large, adjustable neighborhood for classification and can be easily combined with any neural network.

More detail about APPNP can be found in:

Klicpera J , Bojchevski A, S Günnemann. Predict then Propagate: Graph Neural Networks meet Personalized PageRank

- GAT

Graph Attention Networks(GAT) was proposed in 2017 by Petar Veličković et al. By leveraging masked self-attentional layers to address shortcomings of prior graph based method, GAT achieved or matched state of the art performance on both transductive datasets like Cora and inductive dataset like PPI.

More detail about GAT can be found in:

Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2017). Graph attention networks. arXiv preprint arXiv:1710.10903.

This repository contains a implementation of APPNP, GAT and GCN based on MindSpore and GraphLearning

- GCN

Graph Convolutional Networks (GCN) was proposed in 2016 and designed to do semi-supervised learning on graph-structured data. A scalable approach based on an efficient variant of convolutional neural networks which operate directly on graphs was presented. The model scales linearly in the number of graph edges and learns hidden layer representations that encode both local graph structure and features of nodes.

More detail about GCN can be found in:

Thomas N. Kipf, Max Welling. 2016. Semi-Supervised Classification with Graph Convolutional Networks. In ICLR 2016.

## Datasets

The experiment is based on CoraFull.npz, AmazonCoBuy_computers.npz, Coauthor_physics.npz, Coauthor_cs.npz, pubmed_with_mask.npz, cora_v2_with_mask.npz, citeseer_with_mask.npz and reddit_with_mask.npz

## Environment Requirements

- MindSpore >= 1.6.0
- GraphLearning >= 0.1.0

## Quick Start

bash run_bench.sh

## Experiment results

Performance comparison

| Dataset | corav2 | pubmed | citeseet | az_computer | az_photos |
| :----: | :----: | :----: | :----: | :----: | :----: |
| APPNP | 2.25 | 2.61 | 3.37 | 6.78 | 2.77 |
| GAT | 1.57 | 2.96 | 2.34 | 8.65 | 5.03 |
| GCN | 1.31 | 1.97 | 2.05 | 1.94 | 1.46 |