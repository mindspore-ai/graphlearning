// store unit test
import { setActivePinia, createPinia } from "pinia";
import { useMyStore } from "../src/stores/store";
import {
    describe,
    beforeEach,
    beforeAll,
    afterEach,
    afterAll,
    it,
    expect,
} from "vitest";
import type { Dataset } from "../src/types/types.ts";
import * as d3 from "d3";
import { server } from "./mocks/server";
import * as mockDatasets from "./mocks/datasets";
import fetch from "node-fetch";

describe("test get NodesSelectionEntry", () => {
    beforeEach(() => {
        // 创建一个新 pinia，并使其处于激活状态，这样它就会被任何 useStore() 调用自动接收
        // 而不需要手动传递：
        // `useStore(pinia)`
        setActivePinia(createPinia());
    });
    it("should have result by getViewSourceNodesSelectionEntry", () => {
        const myStore = useMyStore();
        expect(
            myStore.getViewSourceNodesSelectionEntry("Topology Space")
        ).toEqual(["full"]);
    });
    it("should have result by getViewTargetNodesSelectionEntry", () => {
        const myStore = useMyStore();
        expect(
            myStore.getViewTargetNodesSelectionEntry("Topology Space")
        ).toEqual(["public", "comparative"]);
    });
});

describe("test fetchOriginDataset", () => {
    //使用 onUnhandledRequest: 'error' 配置服务器可以确保每当有没有相应请求处理程序的请求时都会引发错误。
    beforeAll(() =>
        server
            .listen
            // { onUnhandledRequest: "error" }
            ()
    );

    // Reset any request handlers that we may add during the tests, so they don't affect other tests.
    afterEach(() => server.resetHandlers());

    // Clean up after the tests are finished.
    afterAll(() => server.close());

    beforeEach(() => {
        setActivePinia(createPinia());
    });

    it("should fetch prediction.json & node-embeddings.csv", async () => {
        const myStore = useMyStore();
        global.fetch = fetch;
        const ret = await myStore.fetchOriginDataset(
            mockDatasets.MOCK_URL,
            mockDatasets.MOCK_DATASET_NODE_CLASSIFICATION,
            false
        );
        expect(ret).toBeDefined(); // 确保 ret 定义
        expect(ret).to.be.an("object");
        expect(ret.taskType).toBe("node-classification"); // 验证 taskType
        expect(ret.predLabels).toHaveLength(7);
        expect(ret.embNode).toHaveLength(7);
        console.log(
            "in test should fetch prediction.json & node-embeddings.csv, ret is",
            ret
        );
    });
    it("should error if we don't have prediction.json", async () => {
        const myStore = useMyStore();
        global.fetch = fetch;

        try {
            const ret = await myStore.fetchOriginDataset(
                mockDatasets.MOCK_URL,
                mockDatasets.MOCK_DATASET_LACK_PREDICTION_RESULT,
                false
            );
            // 如果成功执行到这里，表示测试失败
            throw new Error("Expected the function to reject.");
        } catch (error) {
            // 预期函数会拒绝，并且错误消息应该包含特定文本
            console.error(error);
            expect(error.message).toMatch(/404/g);
        }
    });
    it("should error if we don't have node-embeddings.csv", async () => {
        const myStore = useMyStore();
        global.fetch = fetch;

        try {
            const ret = await myStore.fetchOriginDataset(
                mockDatasets.MOCK_URL,
                mockDatasets.MOCK_DATASET_LACK_NODE_EMBEDDINGS,
                false
            );
            // 如果成功执行到这里，表示测试失败
            throw new Error("Expected the function to reject.");
        } catch (error) {
            // 预期函数会拒绝，并且错误消息应该包含特定文本
            console.error(error);
            expect(error.message).toMatch(/404/g);
        }
    });
});
describe("myStore dataset CRUD test", () => {
    beforeEach(() => {
        // 创建一个新 pinia，并使其处于激活状态，这样它就会被任何 useStore() 调用自动接收
        // 而不需要手动传递：
        // `useStore(pinia)`
        setActivePinia(createPinia());
        const myStore = useMyStore();
        myStore.addDataset({
            name: "node-classification0",
            taskType: "node-classification",
            isComplete: true,

            predLabels: [0, 0, 0, 0, 0, 0, 0],
            trueLabels: [0, 2, 0, 1, 3, 0, 1],

            numNodeClasses: 4,
            numGraphs: 1,
        });
    });
    it("should add a dataset", () => {
        const myStore = useMyStore();
        expect(myStore.datasetList.length).toBe(1);
        const mockDatasetNodeClassification: Partial<Dataset> = {
            name: "node-classification1",
            taskType: "node-classification",
            isComplete: true,

            predLabels: [0, 1, 2, 3, 3, 2, 1],
            trueLabels: [1, 2, 0, 1, 3, 0, 1],

            numNodeClasses: 4,

            colorScale: d3
                .scaleOrdinal<string, string>()
                .domain([1, 2, 0, 1, 3, 0, 1].map(String))
                .range(d3.schemeCategory10),

            numGraphs: 1,
        };
        myStore.addDataset(mockDatasetNodeClassification);
        expect(myStore.datasetList.length).toBe(2);
    });
    it("should get a dataset", () => {
        const myStore = useMyStore();
        const getRet = myStore.getDatasetByName("node-classification0");
        expect(getRet).toBeDefined();
    });
    it("should error get a non existing dataset", () => {
        const myStore = useMyStore();
        try {
            const ret = myStore.getDatasetByName("aaa");
            // 如果成功执行到这里，表示测试失败
            throw new Error("Expected the function to reject.");
        } catch (error) {
            // 预期函数会拒绝，并且错误消息应该包含特定文本
            expect(error.message).toContain(
                "Couldn't find Dataset by name: aaa"
            );
        }
    });
    it("should remove dataset", () => {
        const myStore = useMyStore();
        myStore.removeDataset("node-classification0");
        expect(myStore.datasetList).toHaveLength(0);
    });
});
describe("myStore singleDashboard test", () => {
    beforeAll(() => {
        setActivePinia(createPinia());
        const myStore = useMyStore();
        myStore.singleDashboardList = [
            {
                id: "db-test-0",
                refDatasetName: "dataset-test",
                isRoot: true,
                labelType: "true",
                tsneRet: [
                    { id: "0", x: 10, y: 10 },
                    { id: "1", x: 20, y: 20 },
                    { id: "2", x: 30, y: 30 },
                    { id: "3", x: 40, y: 40 },
                    { id: "4", x: 50, y: 50 },
                    { id: "5", x: 60, y: 60 },
                    { id: "6", x: 70, y: 70 },
                ],
            },
        ];
        myStore.recentSingleDashboardList = [
            { id: "db-test-0", renderIndex: 0 },
        ];
    });
    it("should add single dashboard", () => {
        const myStore = useMyStore();
        expect(myStore.singleDashboardList).toHaveLength(1);
        myStore.addSingleDashboard(
            {
                id: "db-test-1",
                refDatasetName: "dataset-test",
            },
            [
                "Topology Space",
                "Latent Space",
                "Topo + Latent Density",
                "Prediction Space",
            ]
        );
        expect(myStore.singleDashboardList).toHaveLength(2);
        expect(myStore.recentSingleDashboardList).toHaveLength(2);
    });
    it("should get single dashboard", () => {
        const myStore = useMyStore();
        const db = myStore.getSingleDashboardById("db-test-0");
        expect(db).toBeDefined();
    });
    it("should error get a non existing single db", () => {
        const myStore = useMyStore();
        try {
            const ret = myStore.getSingleDashboardById("aaa");
            // 如果成功执行到这里，表示测试失败
            throw new Error("Expected the function to reject.");
        } catch (error) {
            // 预期函数会拒绝，并且错误消息应该包含特定文本
            expect(error.message).toContain("Couldn't find");
        }
    });
    it("should to single dashboard", () => {
        const myStore = useMyStore();
        expect(myStore.renderableRecentSingleDashboards.at(-1).id).toEqual(
            "db-test-1"
        );
        myStore.toSingleDashboardById("db-test-0");
        expect(myStore.recentSingleDashboardList.at(-1).id).toEqual(
            "db-test-0"
        );
    });
});
describe("myStore compDashboard test", () => {
    beforeAll(() => {
        setActivePinia(createPinia());
        const myStore = useMyStore();
        myStore.compDashboardList = [
            {
                id: "db-test-0",
                refDatasetsNames: ["dataset-test-0", "dataset-test-1"],
                isRoot: true,
                labelType: "true",
                tsneRet1: [
                    { id: "0", x: 10, y: 10 },
                    { id: "1", x: 20, y: 20 },
                    { id: "2", x: 30, y: 30 },
                    { id: "3", x: 40, y: 40 },
                    { id: "4", x: 50, y: 50 },
                    { id: "5", x: 60, y: 60 },
                    { id: "6", x: 70, y: 70 },
                ],
            },
        ];
        myStore.recentCompDashboardList = [{ id: "db-test-0", renderIndex: 0 }];
    });
    it("should add comp dashboard", () => {
        const myStore = useMyStore();
        expect(myStore.compDashboardList).toHaveLength(1);
        myStore.addCompDashboard(
            {
                id: "db-test-1",
                refDatasetsNames: ["dataset-test-0", "dataset-test-1"],
            },
            [
                "Topology Space",
                "Latent Space",
                "Topo + Latent Density",
                "Prediction Space",
            ]
        );
        expect(myStore.compDashboardList).toHaveLength(2);
        expect(myStore.recentCompDashboardList).toHaveLength(2);
    });
    it("should get comp dashboard", () => {
        const myStore = useMyStore();
        const db = myStore.getCompDashboardById("db-test-0");
        expect(db).toBeDefined();
    });
    it("should error get a non existing comp db", () => {
        const myStore = useMyStore();
        try {
            const ret = myStore.getCompDashboardById("aaa");
            // 如果成功执行到这里，表示测试失败
            throw new Error("Expected the function to reject.");
        } catch (error) {
            // 预期函数会拒绝，并且错误消息应该包含特定文本
            expect(error.message).toContain("Couldn't find");
        }
    });
    it("should to comp dashboard", () => {
        const myStore = useMyStore();
        expect(myStore.renderableRecentCompDashboards.at(-1).id).toEqual(
            "db-test-1"
        );
        myStore.toCompDashboardById("db-test-0");
        expect(myStore.recentCompDashboardList.at(-1).id).toEqual("db-test-0");
    });
});
describe("typed db getter test", () => {
    beforeAll(() => {
        setActivePinia(createPinia());
        const myStore = useMyStore();
        myStore.singleDashboardList = [
            { id: "single-0" },
            { id: "single-1" },
            { id: "single-2" },
        ];
        myStore.compDashboardList = [
            { id: "comp-0" },
            { id: "comp-1" },
            { id: "comp-2" },
        ];
    });
    it("should getTypeReducedDb", () => {
        const myStore = useMyStore();
        const ret = myStore.getTypeReducedDashboardById("comp-0");
        expect(ret).toBeDefined();
        const ret2 = myStore.getTypeReducedDashboardById("single-2");
        expect(ret2).toBeDefined();
        const ret3 = myStore.getTypeReducedDashboardById(ret);
        expect(ret3).toBeDefined;
        const ret4 = myStore.getTypeReducedDashboardById(ret2);
        expect(ret4).toBeDefined;
        try {
            const ret5 = myStore.getTypeReducedDashboardById("aaa");
            // 如果成功执行到这里，表示测试失败
            throw new Error("Expected the function to reject.");
        } catch (error) {
            // 预期函数会拒绝，并且错误消息应该包含特定文本
            expect(error.message).toContain("Neither");
        }
    });
    it("should getTypedDb", () => {
        const myStore = useMyStore();
        const ret = myStore.getTypedDashboardById("comp-0", "comparative");
        expect(ret).toBeDefined();
        const ret2 = myStore.getTypedDashboardById("single-1", "single");
        expect(ret2).toBeDefined();
    });
});
describe("view related test", () => {
    beforeAll(() => {
        setActivePinia(createPinia());
        const myStore = useMyStore();
        myStore.addSingleDashboard(
            {
                id: "db-test-1",
                refDatasetName: "dataset-test",
                fromViewName: "Prediction Space",
            },
            [
                "Topology Space",
                "Latent Space",
                "Topo + Latent Density",
                "Prediction Space",
            ]
        );
    });
    it("should initial a new view", () => {
        const myStore = useMyStore();
        const newView0 = myStore.initialNewView("latent space");
        expect(newView0.meshRadius).toEqual(30);
        const newView1 = myStore.initialNewView("rank view");
        expect(newView1.isBrushEnabled).toBeFalsy();
        const newView2 = myStore.initialNewView("polar view");
        expect(newView2).toHaveProperty("polarData");
        const newView3 = myStore.initialNewView("dense feature view");
        expect(newView3).toHaveProperty("isRelative");
    });
    it("should get view by name", () => {
        const myStore = useMyStore();
        const viewRet = myStore.getViewByName("db-test-1", "Topology Space");
        expect(viewRet).toBeDefined();
    });
    it("should error when get view index by non-existing name", () => {
        const myStore = useMyStore();
        try {
            const viewRet = myStore.getViewByName("db-test-1", "aaa");
            throw new Error("Expected the function to reject.");
        } catch (error) {
            expect(error.message).toBeDefined();
        }
    });
    it("should insert view after name", () => {
        const myStore = useMyStore();
        const len = myStore.singleDashboardList[0].viewsDefinitionList.length;
        const ret0 = myStore.insertViewAfterName(
            "db-test-1",
            "Topology Space",
            myStore.initialNewView("dense feature view")
        );
        expect(myStore.singleDashboardList[0].viewsDefinitionList).toHaveLength(
            len + 1
        );
        expect(ret0).toBeDefined();
    });
    it("should get principal view of db", () => {
        const myStore = useMyStore();
        const principalView = myStore.getPrincipalViewOfDashboard("db-test-1");
        expect(principalView).toBeDefined();
    });
    it("should get snapshot of db", () => {
        const myStore = useMyStore();
        const principalView = myStore.getPrincipalViewOfDashboard("db-test-1");

        principalView.snapshotBase64 = "123123123";
        const base64 = myStore.getPrincipalSnapshotOfDashboard("db-test-1");
        expect(base64).toEqual("123123123");
    });
    it("should remove snapshots", () => {
        const myStore = useMyStore();
        expect(
            myStore.singleDashboardList[0].viewsDefinitionList.some(
                (d) => d.snapshotBase64
            )
        ).toBeTruthy();
        myStore.removeSnapshotsOfDashboard("db-test-1");
        expect(
            myStore.singleDashboardList[0].viewsDefinitionList.every(
                (d) => !d.snapshotBase64
            )
        ).toBeTruthy();
    });
});
describe("test calcAggregatedViewData", () => {
    beforeEach(() => {
        setActivePinia(createPinia());
        const myStore = useMyStore();
        myStore.addSingleDashboard(
            {
                id: "db-test-1",
                refDatasetName: "dataset-test",
            },
            ["Latent Space"]
        );
        const view = myStore.getViewByName("db-test-1", "Latent Space")!;
        view.sourceCoords = [
            { id: "0", x: 10, y: 10 },
            { id: "1", x: 20, y: 20 },
            { id: "2", x: 30, y: 30 },
            { id: "3", x: 40, y: 40 },
            { id: "4", x: 50, y: 50 },
            { id: "5", x: 60, y: 60 },
            { id: "6", x: 70, y: 70 },
        ];
    });
    it("should calcAggregatedViewData", () => {
        //NOTE it's not a pure func, but coupled with data in View
        const myStore = useMyStore();
        const view = myStore.getViewByName("db-test-1", "Latent Space")!;

        expect(view.clusters).toHaveLength(0);
        expect(view.aggregatedLinks).toHaveLength(0);
        expect(view.aggregatedCoords).toHaveLength(0);

        myStore.calcAggregatedViewData(
            view,
            (view) => view.sourceCoords,
            undefined,
            [1, 2, 3, 0, 0, 1, 2],
            true,
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
            ]
        );

        expect(view.clusters.length).toBeGreaterThan(0);
        expect(view.aggregatedLinks.length).toBeGreaterThan(0);
        expect(view.aggregatedCoords.length).toBeGreaterThan(0);
    });
});
describe("test route variables in myStore", () => {
    beforeAll(() => {
        setActivePinia(createPinia());
    });
    it("should set route loading", () => {
        const myStore = useMyStore();
        expect(myStore.routeLoading).toBeUndefined();
        myStore.setRouteLoading();
        expect(myStore.routeLoading).toBeDefined();
    });
    it("should clear route loading", () => {
        const myStore = useMyStore();
        expect(myStore.routeLoading).toBeDefined();
        myStore.clearRouteLoading();
        expect(myStore.routeLoading).toBeUndefined();
    });
});
