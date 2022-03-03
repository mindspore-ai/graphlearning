# Contents

<!--TOC -->

- [Bayesian Graph Collaborative Filtering](#bayesian-graph-collaborative-filtering)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
- [Model Description](#model-description)
    - [Performance](#performance)
- [Description of random situation](#description-of-random-situation)

<!--TOC -->

## [Bayesian Graph Collaborative Filtering](#contents)

Bayesian Graph Collaborative Filtering(BGCF) was proposed in 2020 by Sun J, Guo W, Zhang D et al. By naturally incorporating the
uncertainty in the user-item interaction graph shows excellent performance on Amazon recommendation dataset.This is an example of
training of BGCF with Amazon-Beauty dataset in MindSpore. More importantly, this is the first open source version for BGCF.

[Paper](https://dl.acm.org/doi/pdf/10.1145/3394486.3403254): Sun J, Guo W, Zhang D, et al. A Framework for Recommending Accurate and Diverse Items Using Bayesian Graph Convolutional Neural Networks[C]//Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2020: 2030-2039.

## [Model Architecture](#contents)

Specially, BGCF contains two main modules. The first is sampling, which produce sample graphs based in node copying. Another module
aggregate the neighbors sampling from nodes consisting of mean aggregator and attention aggregator.

## [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

- Dataset size:
  Statistics of dataset used are summarized as below:

  |                    | Amazon-Beauty         |
  | ------------------ | ----------------------|
  | Task               | Recommendation        |
  | # User             | 7068 (1 graph)        |
  | # Item             | 3570                  |
  | # Interaction      | 79506                 |
  | # Training Data    | 60818                 |
  | # Test Data        | 18688                 |  
  | # Density          | 0.315%                |

- Data Preparation
    - Place the dataset to any path you want, the folder should include files as follows(we use Amazon-Beauty dataset as an example)"

  ```python
  .
  └─data
      ├─ratings_Beauty.csv
  ```

    - Generate dataset in mindrecord format for Amazon-Beauty.

  ```builddoutcfg

  cd ./scripts
  # SRC_PATH is the dataset file path you download.
  bash run_process_data_ascend.sh [SRC_PATH]
  ```

## [Environment Requirements](#contents)

- Hardware (GPU)
- Framework
    - [MindSpore >= 1.6.0](https://www.mindspore.cn/install/en)
    - [GraphLearning >= 0.1.0](https://gitee.com/mindspore/docs/tree/master/docs/graphlearning/docs/source_zh_cn)
- For more information, please check the resources below:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)
    - [GraphLearning Tutorials](https://gitee.com/mindspore/docs/tree/master/docs/graphlearning/docs/source_zh_cn)
    - [GraphLearning Python API](https://gitee.com/mindspore/docs/tree/master/docs/graphlearning/api)

## [Quick Start](#contents)

After installing MindSpore via the official website and Dataset is correctly generated, you can start training and evaluation as follows.

- Running on GPU

  ```python
  # run training example with Amazon-Beauty dataset
  bash run_train_gpu.sh 0 dataset_path

  # run evaluation example with Amazon-Beauty dataset
  bash run_eval_gpu.sh 0 dataset_path
  ```  

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```shell
.
└─bgcf
  ├─README.md
  ├─model_utils
  | ├─__init__.py           # Module init file
  | ├─config.py             # Parse arguments
  | ├─device_adapter.py     # Device adapter for ModelArts
  | ├─local_adapter.py      # Local adapter
  | └─moxing_adapter.py     # Moxing adapter for ModelArts
  ├─scripts
  | ├─run_eval_gpu.sh             # Launch evaluation in gpu
  | └─run_train_gpu.sh            # Launch training in gpu
  ├─src
  | ├─bgcf.py              # BGCF model
  | ├─callback.py          # Callback function
  | ├─dataset.py           # Data preprocessing
  | ├─metrics.py           # Recommendation metrics
  | └─utils.py             # Utils for training bgcf
  ├─default_config.yaml    # Configurations file
  ├─eval.py                # Evaluation net
  └─train.py               # Train net
```

### [Script Parameters](#contents)

Parameters for both training and evaluation can be set in default_config.yaml.

- config for BGCF dataset

  ```python
  "learning_rate": 0.001,             # Learning rate
  "num_epoch": 600,                   # Epoch sizes for training
  "num_neg": 10,                      # Negative sampling rate
  "raw_neighs": 40,                   # Num of sampling neighbors in raw graph
  "gnew_neighs": 20,                  # Num of sampling neighbors in sample graph
  "input_dim": 64,                    # User and item embedding dimension
  "l2": 0.03                          # l2 coefficient
  "neighbor_dropout": [0.0, 0.2, 0.3] # Dropout ratio for different aggregation layer
  ```

  default_config.yaml for more configuration.

### [Training Process](#contents)

#### Training

- running on GPU

  ```python
  bash run_train_gpu.sh 0 dataset_path
  ```

  Training result will be stored in the scripts path, whose folder name begins with "train". You can find the result like the
  followings in log.

  ```python
  Epoch 001 iter 12 loss 34660.656
  Epoch 002 iter 12 loss 34129.07
  Epoch 003 iter 12 loss 28200.895
  Epoch 004 iter 12 loss 22179.188
  ```

### [Evaluation Process](#contents)

#### Evaluation

- Evaluation on GPU

  ```python
  bash run_eval_gpu.sh 0 dataset_path
  ```

  Evaluation result will be stored in the scripts path, whose folder name begins with "eval". You can find the result like the
  followings in log.

  ```python
  epoch:680,  recall_@10:0.10245,  recall_@20:0.15472,  ndcg_@10:0.07381,  ndcg_@20:0.09144,  
  sedp_@10:0.01904,  sedp_@20:0.01526,  nov_@10:7.60682,  nov_@20:7.81804
  ```

## [Model Description](#contents)

### [Performance](#contents)

#### Training Performance

| Parameter                      | BGCF GPU                                   |
| ------------------------------ | ------------------------------------------ |
| Model Version                  | Inception V1                               |
| Resource                       | Tesla V100-PCIE                            |
| uploaded Date                  | 10/30/2021(month/day/year)                 |
| MindSpore Version              | 1.4.1                                      |
| Dataset                        | Amazon-Beauty                              |
| Training Parameter             | epoch=680,steps=12,batch_size=5000,lr=0.001|
| Optimizer                      | Adam                                       |
| Loss Function                  | BPR loss                                   |
| Training Cost                  | 15min                                      |
| Scripts                        | [bgcf script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/gnn/bgcf) |

#### Evaluation Performance

| Parameter                      | BGCF GPU                     |
| ------------------------------ | ---------------------------- |
| Model Version                  | Inception V1                 |
| Resource                       | Tesla V100-PCIE              |
| uploaded Date                  | 10/30/2021(month/day/year)   |
| MindSpore Version              | 1.4.1                        |
| Dataset                        | Amazon-Beauty                |
| Batch_size                     | 5000                         |
| Output                         | probability                  |
| Recall@20                      | 0.15472                      |
| NDCG@20                        | 0.09144                      |

## [Description of random situation](#contents)

BGCF model contains lots of dropout operations, if you want to disable dropout, set the neighbor_dropout to [0.0, 0.0, 0.0] in default_config.yaml.
