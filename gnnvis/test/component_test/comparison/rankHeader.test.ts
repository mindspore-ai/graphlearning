import { createTestingPinia } from "@pinia/testing";
import ElementPlus from "element-plus";
import "element-plus/dist/index.css";

import { describe, it, expect, vi } from "vitest";

import { mount } from "@vue/test-utils";

import RankHeader from "../../../src/components/comparison/rankHeader.vue";
import {
    CompDashboard,
    Dataset,
    RankDatum,
    RankView,
} from "../../../src/types/types";

const props: InstanceType<typeof RankHeader>["$props"] = {
    dbId: "db-test",
    viewName: "Comparative Rank View",
};
const testPinia = createTestingPinia({
    // createSpy: vi.fn,
    stubActions: false,
    initialState: {
        my: {
            // ...myStore,

            datasetList: [
                {
                    name: "dataset-test-0",
                    taskType: "node-classification",
                    predLabels: [1, 2, 3, 0, 3, 2, 1],
                    trueLabels: [0, 2, 1, 3, 0, 2, 2],
                } as Partial<Dataset>,
                {
                    name: "dataset-test-1",
                    taskType: "node-classification",
                    predLabels: [1, 1, 2, 2, 3, 3, 3],
                    trueLabels: [0, 0, 1, 1, 0, 0, 2],
                } as Partial<Dataset>,
            ],
            compDashboardList: [
                {
                    id: "db-test",
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
                    tsneRet2: [
                        { id: "0", x: 11, y: 11 },
                        { id: "1", x: 21, y: 21 },
                        { id: "2", x: 31, y: 31 },
                        { id: "3", x: 41, y: 41 },
                        { id: "4", x: 51, y: 51 },
                        { id: "5", x: 61, y: 61 },
                        { id: "6", x: 71, y: 71 },
                    ],
                    isHighlightCorrespondingNode: true,
                    nodesSelections: {
                        comparativeOut: {},
                        rankOut: {},
                        comparative: {
                            "0": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "1": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "2": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "3": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "4": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "5": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "6": { gid: "0", parentDbIndex: 0, hop: -1 },
                        },
                        full: {
                            "0": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "1": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "2": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "3": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "4": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "5": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "6": { gid: "0", parentDbIndex: 0, hop: -1 },
                        },
                        public: {
                            "0": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "1": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "2": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "3": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "4": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "5": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "6": { gid: "0", parentDbIndex: 0, hop: -1 },
                        },
                    },
                    rankWorker: {
                        workerFn: () =>
                            new Promise<void>((resolve, reject) => {
                                resolve();
                            }),
                        workerStatus: "ERROR",
                        workerTerminate: () => {},
                    },
                    viewsDefinitionList: [
                        {
                            viewName: props.viewName,
                            // headerComp: shallowRef(null),
                            // bodyComp: shallowRef(null),
                            initialWidth: 500,
                            initialHeight: 541,

                            bodyHeight: 500,
                            bodyWidth: 500,
                            bodyMargins: {
                                top: 0.05,
                                right: 0.05,
                                bottom: 0.05,
                                left: 0.05,
                            },

                            resizeEndSignal: false,
                            isBrushEnabled: true,
                            brushEnableFunc: () => {},
                            brushDisableFunc: () => {},
                            panEnableFunc: () => {},
                            panDisableFunc: () => {},
                            hideRectWhenClearSelFunc: () => {},
                            resetZoomFunc: () => {}, //NOTE 在view组件挂在之后更新
                            isGettingSnapshot: false,
                            gettingSnapShotError: undefined,
                            snapshotBase64: "",

                            bodyProps: {},
                            headerProps: {},

                            setAttr(attr, value: any) {
                                this[attr] = value;
                                return this;
                            },
                            rankData: [
                                {
                                    id: "0",
                                    r1: 0,
                                    r2: 1,
                                    d1: 0.1,
                                    d2: 0.2,
                                },
                                {
                                    id: "1",
                                    r1: 1,
                                    r2: 0,
                                    d1: 0.2,
                                    d2: 0.1,
                                },
                                {
                                    id: "2",
                                    r1: 2,
                                    r2: 3,
                                    d1: 0.3,
                                    d2: 0.4,
                                },
                                {
                                    id: "3",
                                    r1: 3,
                                    r2: 2,
                                    d1: 0.4,
                                    d2: 0.3,
                                },
                                {
                                    id: "4",
                                    r1: 4,
                                    r2: 5,
                                    d1: 0.4,
                                    d2: 0.5,
                                },
                                {
                                    id: "5",
                                    r1: 5,
                                    r2: 4,
                                    d1: 0.5,
                                    d2: 0.4,
                                },
                                {
                                    id: "6",
                                    r1: 6,
                                    r2: 6,
                                    d1: 0.8,
                                    d2: 0.7,
                                },
                            ] as Array<RankDatum>,
                            rankEmbDiffAlgo: "center",
                        } as RankView,
                    ],
                } as Partial<CompDashboard>,
            ],
        },
    },
});

describe("test rank header", () => {
    it("test render", async () => {
        const wrapper = mount(RankHeader, {
            props: props,
            global: {
                plugins: [testPinia, ElementPlus],
            },
        });
        const switcher = wrapper.find(".el-switch");
        expect(switcher.classes()).toContain("is-disabled"); //NOTE worker status模拟为ERROR
        wrapper.unmount();
    });
});
