link_pred_res = {
    "trueAllowEdges": [
        [0, 1], [0, 2]
    ],
    "falseAllowEdges": [
        [1, 2], [2, 3]
    ],
    "trueUnseenTopK": 1,
    "trueUnseenEdgesSorted": {
        3: [1]
    }
}

node_classify_res = {
    "numNodeClasses": 4,
    "predLabels": [
        'O', 'C', 'C', 'H', 'F'
    ],
    "trueLabels": [
        'C', 'O', 'O', 'H', 'F'
    ],
}

graph_classify_res = {
    "numGraphClasses": 2,
    "graph_index": [
        "0", "1"
    ],
    "predLabels": [
        "OC2", "HF"
    ],
    "trueLabels": [
        "CO2", "HF"
    ],
    "phaseDict": {
        '0': "train",
        '1': "valid",
        '2': "predict",
    },
    "phase": [
        0, 2
    ]
}
