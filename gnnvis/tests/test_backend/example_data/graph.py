graph_dict = {
    "directed": False,
    "multigraph": True,
    "graphs": {
        "0": {
            "id": 0,
            "label": 'CO2',
            "nodes": [0, 1, 2],
            "edges": [0, 1],
        },
        "1": {
            "id": 1,
            "label": 'HF',
            "nodes": [3, 4],
            "edges": [2],
        },
    },
    "nodes": [
        {"id": 0, "label": 'C'},
        {"id": 1, "label": 'O'},
        {"id": 2, "label": 'O'},
        {"id": 3, "label": 'H'},
        {"id": 4, "label": 'F'},
    ],
    "edges": [
        {"eid": 0, "source": 0, "target": 1, "label": "C-O"},
        {"eid": 1, "source": 0, "target": 2, "label": "C-O"},
        {"eid": 2, "source": 3, "target": 4, "label": "H-F"},
    ],
    "isDenseNodeFeature": False,
}
