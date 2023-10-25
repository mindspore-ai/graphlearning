import numpy as np
import json

data = np.loadtxt("node-dense-features.csv", delimiter=",")

node_cnt = len(data)
node_feature_size = len(data[0])

res = {
    "numNodeFeatureDims": node_feature_size,
    "nodeFeatureIndexes": [[] for _ in range(node_cnt)],
    "nodeFeatureValues": [[] for _ in range(node_cnt)]
}

for node_id, node_feature in enumerate(data):
    for feature_id, feature in enumerate(node_feature):
        if feature > 0:
            res["nodeFeatureIndexes"][node_id].append(feature_id)
            res["nodeFeatureValues"][node_id].append(feature)

# save res to json file
with open("node-sparse-features.json", "w") as f:
    json.dump(res, f)
