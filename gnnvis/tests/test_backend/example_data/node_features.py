import numpy as np


node_dense_features = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
node_dense_features_name = ["t1", "t2", "t3"]

node_sparse_features = {
    "numNodeFeatureDims": 888,
    "nodeFeatureIndexes": [
        [65, 77, 391, 801],
        [30, 102, 887],
    ],
    "nodeFeatureValues": [
        [1.2, 1.5, 1.8, 0.3],
        [0.2, 1.4, 1.5],
    ]
}
