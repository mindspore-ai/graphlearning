import { describe, beforeEach, test, it, expect, beforeAll } from "vitest";
import * as graphUtils from "../src/utils/graphUtils";

describe("test graphUtils", () => {
    it("should calc correct isEmptyDict", () => {
        expect(graphUtils.isEmptyDict({ a: 1, b: 2 })).toBeFalsy();
        expect(graphUtils.isEmptyDict({})).toBeTruthy();
    });

    it("should calc correct nodeMapGraph2GraphMapNode", () => {
        const ret = graphUtils.nodeMapGraph2GraphMapNodes({
            0: {
                gid: 0,
            },
            1: { gid: 1 },
            2: {
                gid: 0,
            },
            3: { gid: 1 },
        });
        console.log(ret);
        expect(ret).toEqual({
            "0": { "0": { gid: 0 }, "2": { gid: 0 } },
            "1": { "1": { gid: 1 }, "3": { gid: 1 } },
        });
    });
    it("should calc correct graphMapNodes2nodeMapGraph", () => {
        const ret = graphUtils.graphMapNodes2nodeMapGraph({
            "0": { 0: {}, 1: {}, 2: {} },
            "1": { 3: {}, 4: {}, 5: {} },
        });
        console.log(ret);
        expect(ret).toEqual({
            "0": { gid: "0" },
            "1": { gid: "0" },
            "2": { gid: "0" },
            "3": { gid: "1" },
            "4": { gid: "1" },
            "5": { gid: "1" },
        });
    });
    it("should calc correct computeNeighborMasks", () => {
        const ret = graphUtils.computeNeighborMasks(
            5,
            [
                [{ nid: "1", eid: "0" }],
                [
                    { nid: "2", eid: "1" },
                    { nid: "3", eid: "2" },
                ],
                [],
                [{ nid: "4", eid: "3" }],
                [],
            ],
            2
        );
        console.log(ret);
        expect(ret.neighborMasks).toEqual(["f", "1f", "e", "1e", "18"]);
        expect(ret.neighborMasksByHop).toEqual([
            ["3", "f", "6", "1a", "18"],
            ["e", "1e", "c", "1c", "10"],
        ]);
        expect(ret.neighborMasksByHopPure).toEqual([
            ["3", "f", "6", "1a", "18"],
            ["c", "11", "a", "6", "8"],
        ]);
    });

    it("should calc correct calcNeighborDict", () => {
        const ret = graphUtils.calcNeighborDict(["2", "3"], 2, [
            ["3", "f", "6", "1a", "18"],
            ["e", "1e", "c", "1c", "10"],
        ]);
        console.log(ret);
        expect(ret).toEqual({
            "1": { hop: 0 },
            "2": { hop: -1 },
            "3": { hop: -1 },
            "4": { hop: 0 },
        });
    });
    it("should calc correct filterEdgeAndComputeDict", () => {
        const ret = graphUtils.filterEdgeAndComputeDict(5, [
            { source: 0, target: 1 },
            { source: 1, target: 0 },
            { source: 1, target: 1 },
            { source: 1, target: 2 },
            { source: 2, target: 3 },
            { source: 4, target: 3 },
        ]);
        console.log(ret);
        expect(ret.edges).toEqual([
            { source: "0", target: "1", eid: "0" },
            { source: "1", target: "2", eid: "1" },
            { source: "2", target: "3", eid: "2" },
            { source: "4", target: "3", eid: "3" },
        ]);
        expect(ret.nodeMapLink).toEqual([
            [{ nid: "1", eid: "0" }],
            [
                { nid: "0", eid: "0" },
                { nid: "2", eid: "1" },
            ],
            [
                { nid: "1", eid: "1" },
                { nid: "3", eid: "2" },
            ],
            [
                { nid: "2", eid: "2" },
                { nid: "4", eid: "3" },
            ],
            [{ nid: "3", eid: "3" }],
        ]);
    });
    it("should calc correct filterEdgesAndComputeDictInMultiGraph", () => {
        const ret = graphUtils.filterEdgesAndComputeDictInMultiGraph(
            [
                {
                    id: "graph0",
                    label: 1,
                    nodes: ["0", "1", "2"],
                    edges: ["0", "1", "2", "3"],
                },
                {
                    id: "graph1",
                    label: 1,
                    nodes: ["3", "4", "5"],
                    edges: ["4", "5", "6", "7"],
                },
            ],
            6,
            {
                0: { source: 0, target: 1 },
                1: { source: 1, target: 0 },
                2: { source: 1, target: 2 },

                3: { source: 2, target: 0 },
                4: { source: 3, target: 4 },
                5: { source: 4, target: 4 },
                6: { source: 4, target: 5 },
                7: { source: 3, target: 5 },
            }
        );
        expect(ret.filteredGraphArr).toEqual([
            {
                id: "graph0",
                label: 1,
                nodes: ["0", "1", "2"],
                edges: ["0", "1", "2", "3"],
                gid: "0",
                links: ["0", "1", "2"],
            },
            {
                id: "graph1",
                label: 1,
                nodes: ["3", "4", "5"],
                edges: ["4", "5", "6", "7"],
                gid: "1",
                links: ["3", "4", "5"],
            },
        ]);
        expect(ret.filteredGraphRecord).toEqual({
            "0": {
                nodesRecord: { "0": "0", "1": "0", "2": "0" },
                linksRecord: { "0": "0", "1": "0", "2": "0" },
            },
            "1": {
                nodesRecord: { "3": "1", "4": "1", "5": "1" },
                linksRecord: { "3": "1", "4": "1", "5": "1" },
            },
        });
        expect(ret.filteredEdges).toEqual([
            { source: "0", target: "1", eid: "0" },
            { source: "1", target: "2", eid: "1" },
            { source: "2", target: "0", eid: "2" },
            { source: "3", target: "4", eid: "3" },
            { source: "4", target: "5", eid: "4" },
            { source: "3", target: "5", eid: "5" },
        ]);
        expect(ret.nodeMapLink).toEqual([
            [
                { nid: "1", eid: "0" },
                { nid: "2", eid: "2" },
            ],
            [
                { nid: "0", eid: "0" },
                { nid: "2", eid: "1" },
            ],
            [
                { nid: "1", eid: "1" },
                { nid: "0", eid: "2" },
            ],
            [
                { nid: "4", eid: "3" },
                { nid: "5", eid: "5" },
            ],
            [
                { nid: "3", eid: "3" },
                { nid: "5", eid: "4" },
            ],
            [
                { nid: "4", eid: "4" },
                { nid: "3", eid: "5" },
            ],
        ]);
    });
    it("should calc calcGraphCoords", () => {
        const ret = graphUtils.calcGraphCoords(
            [{ id: "0" }, { id: "1" }, { id: "2" }, { id: "3" }],
            [
                { eid: "0", source: "0", target: "1" },
                { eid: "1", source: "1", target: "2" },
                { eid: "2", source: "1", target: "3" },
            ]
        );
        // console.log(ret);
        expect(ret).toHaveLength(4);
        expect(
            ret.every((d) => Object.hasOwn(d, "x") && Object.hasOwn(d, "y"))
        ).toBeTruthy();
    });
    it("should calc correct calcSeparatedMultiGraphCoords", () => {
        const ret = graphUtils.calcSeparatedMultiGraphCoords(
            [
                {
                    gid: "0",
                    nodes: ["0", "1", "2"],
                    links: [
                        { eid: "0", source: "0", target: "1" },
                        { eid: "1", source: "1", target: "2" },
                    ],
                },
                {
                    gid: "1",
                    nodes: ["3", "4", "5"],
                    links: [
                        { eid: "2", source: "3", target: "4" },
                        { eid: "3", source: "4", target: "5" },
                    ],
                },
            ],
            6
        );
        console.log(ret);
        expect(
            ret.every((d) => Object.hasOwn(d, "x") && Object.hasOwn(d, "y"))
        ).toBeTruthy();
    });

    it("should calc correct rescaleCoords", () => {
        const ret = graphUtils.rescaleCoords(
            [
                { id: "0", x: 10, y: 10 },
                { id: "4", x: 40, y: 40 },
                { id: "6", x: 70, y: 70 },
            ],
            [0, 1000],
            [0, 1000]
        );
        console.log(ret);
        expect(ret).toEqual([
            { id: "0", x: 0, y: 0 },
            { id: "4", x: 500, y: 500 },
            { id: "6", x: 1000, y: 1000 },
        ]);
    });

    it("should calc correct rescalePolarCoords", () => {
        const ret = graphUtils.rescalePolarCoords(
            [
                {
                    id: "0",
                    angle: 0.5,
                    radius: 1,
                },
                {
                    id: "1",
                    angle: 0.2,
                    radius: 2,
                },
                {
                    id: "2",
                    angle: 0.8,
                    radius: 3,
                },
            ],
            undefined,
            undefined,
            () => [0, 10],
            undefined,
            () => 1
        );
        console.log(ret);
        expect(ret).toHaveLength(3);
        expect(ret[0].angle).toBeCloseTo(Math.PI / 2, 4);
        expect(ret[1].radius).toBeCloseTo((Math.sqrt(2) / 2) * 10);
        expect(ret[2].angle).toBeCloseTo(Math.PI, 4);
    });

    it("should calc correct calcTsne", () => {
        const ret = graphUtils.calcTsne([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [1.1, 1.3, 1.5],
            [2.2, 2.6, 2.9],
            [3.1, 3.9, 3.8],
            [10.5, 10.3, 9.3],
        ]);
        console.log(ret);
        expect(ret).toHaveLength(7);
        expect(ret.every((d) => d.length === 2)).toBeTruthy();
    });
    it("should calc correct calcVectorDist", () => {
        const ret = graphUtils.calcVectorDist(
            { "0": true, "1": true },
            { "4": true, 5: "true", 6: "true" },
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [1.1, 1.3, 1.5],
                [2.2, 2.6, 2.9],
                [3.1, 3.9, 3.8],
                [10.5, 10.3, 9.3],
            ]
        );
        console.log(ret);
        expect(ret).toHaveLength(2);
        expect(ret[0].dist).toBeCloseTo(0.0301, 3);
        expect(ret[1].dist).toBeCloseTo(0.0043, 3);
    });
    it("should calc correct calcRank3", () => {
        const ret = graphUtils.calcRank3(
            "single",
            "0",
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            [
                [11, 22, 33],
                [44, 55, 66],
                [77, 88, 99],
            ]
        );
        console.log(ret);
        expect(ret).toHaveLength(3);
        expect(ret[0].r1).toEqual(0);
        expect(ret[0].r2).toEqual(0);
        const ret2 = graphUtils.calcRank3(
            "center",
            ["0", "1"],
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            [
                [11, 22, 33],
                [44, 55, 66],
                [77, 88, 99],
            ]
        );
        console.log(ret2);
        expect(ret2).toHaveLength(3);
        expect(ret2[0].r1).toEqual(2);
        expect(ret2[0].r2).toEqual(2);
        const ret3 = graphUtils.calcRank3(
            "average",
            ["0", "1"],
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            [
                [11, 22, 33],
                [44, 55, 66],
                [77, 88, 99],
            ]
        );
        console.log(ret3);
        expect(ret3).toHaveLength(3);
        expect(ret3[0].r1).toEqual(0);
        expect(ret3[0].r2).toEqual(0);
    });
    it("should calc correct calcPolar", () => {
        const ret = graphUtils.calcPolar(
            "hamming",
            "center",
            ["0", "1"],
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [1.1, 2.3, 5.5],
                [6.1, 6.1, 0.3],
            ],
            [
                [11, 22, 33],
                [44, 55, 66],
                [77, 88, 99],
                [11, 33, 77],
                [33, 27, 92],
            ],
            [
                [{ nid: "1", eid: "0" }],
                [
                    { nid: "2", eid: "1" },
                    { nid: "3", eid: "2" },
                ],
                [],
                [{ nid: "4", eid: "3" }],
                [],
            ],
            2,
            [
                ["3", "f", "6", "1a", "18"],
                ["e", "1e", "c", "1c", "10"],
            ]
        );
        console.log(ret);
        expect(ret).toHaveLength(3);
        const ret2 = graphUtils.calcPolar(
            "jaccard",
            "single",
            "0",
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [1.1, 2.3, 5.5],
                [6.1, 6.1, 0.3],
            ],
            [
                [11, 22, 33],
                [44, 55, 66],
                [77, 88, 99],
                [11, 33, 77],
                [33, 27, 92],
            ],
            [
                [{ nid: "1", eid: "0" }],
                [
                    { nid: "2", eid: "1" },
                    { nid: "3", eid: "2" },
                ],
                [],
                [{ nid: "4", eid: "3" }],
                [],
            ],
            2,
            [
                ["3", "f", "6", "1a", "18"],
                ["e", "1e", "c", "1c", "10"],
            ]
        );
        console.log(ret2);
        expect(ret2).toHaveLength(3);
    });
    it("should calc correct calcHexbinClusters", () => {
        const ret = graphUtils.calcHexbinClusters(
            [
                { id: "0", x: 10, y: 10, data: 0 },
                { id: "1", x: 20, y: 20, data: 0 },
                { id: "2", x: 30, y: 30, data: 1 },
                { id: "3", x: 40, y: 40, data: 1 },
                { id: "4", x: 50, y: 50, data: 2 },
                { id: "5", x: 60, y: 60, data: 2 },
                { id: "6", x: 70, y: 70, data: 3 },
            ],
            () => [
                [0, 0],
                [100, 100],
            ],
            undefined,
            undefined,
            30,
            (d) => d.data
        );
        console.log(ret);
        expect(ret.clusters).toHaveLength(4);
    });
    it("should calc correct calcAggregatedLinks", () => {
        const ret = graphUtils.calcAggregatedLinks(
            [["0"], ["1", "2", "3", "4"], ["5"], ["6"]],
            [
                { eid: "0-1", source: 0, target: 1 },
                { eid: "0-2", source: 0, target: 2 },
                { eid: "0-3", source: 0, target: 3 },
                { eid: "0-4", source: 0, target: 4 },
                { eid: "0-5", source: 0, target: 5 },
                { eid: "0-6", source: 0, target: 6 },
                { eid: "1-2", source: 1, target: 2 },
                { eid: "1-3", source: 1, target: 3 },
                { eid: "1-4", source: 1, target: 4 },
                { eid: "1-5", source: 1, target: 5 },
                { eid: "1-6", source: 1, target: 6 },
                { eid: "2-3", source: 2, target: 3 },
                { eid: "2-4", source: 2, target: 4 },
                { eid: "2-5", source: 2, target: 5 },
                { eid: "2-6", source: 2, target: 6 },
                { eid: "3-4", source: 3, target: 4 },
                { eid: "3-5", source: 3, target: 5 },
                { eid: "3-6", source: 3, target: 6 },
                { eid: "4-5", source: 4, target: 5 },
                { eid: "4-6", source: 4, target: 6 },
                { eid: "5-6", source: 5, target: 6 },
            ],
            (id) => {
                const dict = {
                    "0": "0",
                    "1": "1",
                    "2": "1",
                    "3": "1",
                    "4": "1",
                    "5": "2",
                    "6": "3",
                };
                return dict[id];
            }
        );
        console.log(ret);
        expect(ret).toHaveLength(7);
    });
});
