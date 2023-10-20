import { rest } from "msw";
import * as mockDatasets from "./datasets";
import { NodeClassificationPredData } from "../../src/types/types";
import { baseUrl } from "../../src/api/api";

export const handlers = [
    //正常的node-classification
    rest.get(
        `${mockDatasets.MOCK_URL}${mockDatasets.MOCK_DATASET_NODE_CLASSIFICATION}/prediction-results.json`,
        //注意，这个地方没有斜杠，因为MOCK_URL本身有斜杠了
        (req, res, ctx) => {
            return res(
                ctx.status(200),
                ctx.json({
                    numNodeClasses: 4,
                    predLabels: [1, 2, 3, 0, 3, 2, 1],
                    trueLabels: [0, 2, 1, 3, 0, 2, 2],
                    taskType: "node-classification",
                } as NodeClassificationPredData)
            );
        }
    ),
    rest.get(
        `${mockDatasets.MOCK_URL}${mockDatasets.MOCK_DATASET_NODE_CLASSIFICATION}/node-embeddings.csv`,
        (req, res, ctx) => {
            return res(
                ctx.status(200),
                ctx.text(
                    "1, 2, 3\n4, 5, 6\n 7, 8, 9\n1.1, 1.3, 1.5\n2.2, 2.6, 2.9\n3.1, 3.9, 3.8\n10.5, 10.3, 9.3\n"
                )
            );
        }
    ),
    rest.get(
        `${mockDatasets.MOCK_URL}${mockDatasets.MOCK_DATASET_NODE_CLASSIFICATION}/graph.json`,

        (req, res, ctx) => {
            return res(
                ctx.status(200),
                ctx.json({
                    multigraph: false,
                    graphs: {},
                    nodes: [
                        { id: "0" },
                        { id: "1" },
                        { id: "2" },
                        { id: "3" },
                        { id: "4" },
                        { id: "5" },
                        { id: "6" },
                    ],
                    edges: [
                        { eid: "0", source: "0", target: "1" },
                        { eid: "1", source: "1", target: "2" },
                        { eid: "2", source: "2", target: "3" },
                        { eid: "3", source: "3", target: "4" },
                        { eid: "4", source: "4", target: "5" },
                        { eid: "5", source: "5", target: "6" },
                    ],
                })
            );
        }
    ),

    //没有prediction-results.json文件
    rest.get(
        `${mockDatasets.MOCK_URL}${mockDatasets.MOCK_DATASET_LACK_PREDICTION_RESULT}/prediction-results.json`,
        (req, res, ctx) => {
            return res(ctx.status(404));
        }
    ),

    //有prediction-results.json文件, 但是没有node-embeddings.csv文件
    rest.get(
        `${mockDatasets.MOCK_URL}${mockDatasets.MOCK_DATASET_LACK_NODE_EMBEDDINGS}/prediction-results.json`,
        (req, res, ctx) => {
            return res(
                ctx.status(200),
                ctx.json({
                    numNodeClasses: 4,
                    predLabels: [1, 2, 3, 0, 3, 2, 1],
                    trueLabels: [0, 2, 1, 3, 0, 2, 2],
                    taskType: "node-classification",
                } as NodeClassificationPredData)
            );
        }
    ),
    rest.get(
        `${mockDatasets.MOCK_URL}${mockDatasets.MOCK_DATASET_LACK_NODE_EMBEDDINGS}/node-embeddings.csv`,
        (req, res, ctx) => {
            return res(ctx.status(404));
        }
    ),

    //硬编码url的graphFeature
    rest.get(
        `${baseUrl}test-graph-feature-dataset/graph-custom-index.json`,
        (req, res, ctx) => {
            console.log(
                "now mock fetch ",
                `${baseUrl}test-graph-feature-dataset/graph-custom-index.json`
            );
            return res(
                ctx.status(200),
                ctx.json({
                    index_target: "graph",
                    number_of_C: {
                        "0": 14,
                        "1": 9,
                        "2": 9,
                        "3": 16,
                        "4": 6,
                        "5": 16,
                        "6": 13,
                    },
                    number_of_F: {
                        "0": 0,
                        "1": 0,
                        "2": 0,
                        "3": 0,
                        "4": 2,
                        "5": 0,
                        "6": 0,
                    },
                    number_of_NH2: {
                        "0": 1,
                        "1": 1,
                        "2": 1,
                        "3": 1,
                        "4": 1,
                        "5": 4,
                        "6": 1,
                    },
                })
            );
        }
    ),
];
