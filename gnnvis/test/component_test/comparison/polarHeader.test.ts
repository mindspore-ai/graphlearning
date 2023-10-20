import { createTestingPinia } from "@pinia/testing";
import { useMyStore } from "../../../src/stores/store";
import ElementPlus from "element-plus";
import "element-plus/dist/index.css";

import { describe, it, expect, vi } from "vitest";

import { mount } from "@vue/test-utils";

import PolarHeader from "../../../src/components/comparison/polarHeader.vue";
import {
    CompDashboard,
    Dataset,
    PolarDatum,
    PolarView,
} from "../../../src/types/types";

const props: InstanceType<typeof PolarHeader>["$props"] = {
    dbId: "db-test",
    viewName: "Comparative Polar View",
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
                    srcLinksArr: [
                        { eid: "0", source: "0", target: "1", gid: "0" },
                        { eid: "1", source: "1", target: "2", gid: "0" },
                        { eid: "2", source: "2", target: "3", gid: "0" },
                        { eid: "3", source: "3", target: "4", gid: "0" },
                        { eid: "4", source: "4", target: "5", gid: "0" },
                        { eid: "5", source: "5", target: "6", gid: "0" },
                    ],
                    isHighlightCorrespondingNode: true,
                    nodesSelections: {
                        comparativeOut: {},
                        full: {
                            "0": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "1": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "2": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "3": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "4": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "5": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "6": { gid: "0", parentDbIndex: 0, hop: -1 },
                        },
                        comparative: {
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
                    polarWorker: {
                        workerFn: () =>
                            new Promise<void>((resolve, reject) => {
                                resolve();
                            }),
                        workerStatus: "SUCCESS",
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
                            isShowAggregation: false,
                            polarData: [
                                {
                                    id: "0",
                                    hop: 0,
                                    embDiff: 0.1,
                                    topoDist: 0.3,
                                },
                                {
                                    id: "1",
                                    hop: 0,
                                    embDiff: 0.2,
                                    topoDist: 0.4,
                                },
                                {
                                    id: "2",
                                    hop: 0,
                                    embDiff: 0.3,
                                    topoDist: 0.5,
                                },
                                {
                                    id: "3",
                                    hop: 0,
                                    embDiff: 0.4,
                                    topoDist: 0.6,
                                },
                                {
                                    id: "4",
                                    hop: 1,
                                    embDiff: 0.5,
                                    topoDist: 0.2,
                                },
                                {
                                    id: "5",
                                    hop: 1,
                                    embDiff: 0.3,
                                    topoDist: 0.7,
                                },
                                {
                                    id: "6",
                                    hop: 1,
                                    embDiff: 0.5,
                                    topoDist: 0.8,
                                },
                            ] as Array<PolarDatum>, //[{id,hop,embDiff,topoDist}, {id,hop,embDiff,topoDist}, ...]
                            polarEmbDiffAlgo: "center",
                            polarTopoDistAlgo: "hamming",
                            hops: 2,
                        } as PolarView,
                    ],
                } as Partial<CompDashboard>,
            ],
        },
    },
});

describe("test polar header", () => {
    it("test render", () => {
        const myStore = useMyStore();
        expect(
            myStore.compDashboardList[0].viewsDefinitionList[0].linkOpacity
        ).toBeUndefined();
        const wrapper = mount(PolarHeader, {
            props: props,
            global: {
                plugins: [testPinia, ElementPlus],
            },
        });
        expect(
            myStore.compDashboardList[0].viewsDefinitionList[0].linkOpacity
        ).toBe(0.2);

        wrapper.unmount();
    });
});
