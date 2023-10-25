import { createTestingPinia } from "@pinia/testing";
import { useMyStore } from "../../../src/stores/store";
import ElementPlus from "element-plus";
import "element-plus/dist/index.css";
import * as d3 from "d3";
import { describe, it, expect } from "vitest";
import { mount } from "@vue/test-utils";

import SparseFeatureHeader from "../../../src/components/publicViews/sparseFeatureHeader.vue";
import { Dataset, SingleDashboard, View } from "../../../src/types/types";

const props: InstanceType<typeof SparseFeatureHeader>["$props"] = {
    dbId: "db-test",
    viewName: "Feature Space",
};

const testPinia = createTestingPinia({
    // createSpy: vi.fn,
    stubActions: false,
    initialState: {
        my: {
            // ...myStore,

            datasetList: [
                {
                    name: "dataset-test",
                    taskType: "node-classification",
                    predLabels: [1, 2, 3, 0, 3, 2, 1],
                    trueLabels: [0, 2, 1, 3, 0, 2, 2],
                    numNodeClasses: 4,
                    colorScale: d3
                        .scaleOrdinal<string, string>()
                        .domain([1, 2, 0, 1, 3, 0, 1].map(String))
                        .range(d3.schemeCategory10),

                    nodeSparseFeatureIndexes: [
                        [999, 888, 777],
                        [102, 22, 134, 1532],
                        [1142, 843, 1531],
                        [111, 555, 666],
                        [999, 888, 777],
                        [234, 983, 719],
                        [1133, 1251, 1932],
                    ],
                    nodeSparseFeatureValues: [
                        [0.1, 0.2, 0.3],
                        [0.9, 0.8, 0.3],
                        [1.2, 1.3, 1.4],
                        [1.2, 2.3, 1.4],
                        [2.2, 1.3, 1.4],
                        [1.2, 1.5, 2.4],
                        [0.7, 1.2, 0.3],
                    ],
                    numNodeSparseFeatureDims: 2000,
                } as Partial<Dataset>,
            ],
            singleDashboardList: [
                {
                    id: "db-test",
                    refDatasetName: "dataset-test",
                    isRoot: true,
                    labelType: "true",
                    isHighlightCorrespondingNode: true,
                    parentId: "0",
                    nodesSelections: {
                        full: {
                            "0": {
                                gid: "0",
                                parentDbIndex: 0,
                                hop: -1,
                            },
                            "1": {
                                gid: "0",
                                parentDbIndex: 0,
                                hop: -1,
                            },
                            "2": {
                                gid: "0",
                                parentDbIndex: 0,
                                hop: -1,
                            },
                            "3": {
                                gid: "0",
                                parentDbIndex: 0,
                                hop: -1,
                            },
                            "4": {
                                gid: "0",
                                parentDbIndex: 0,
                                hop: -1,
                            },
                            "5": {
                                gid: "0",
                                parentDbIndex: 0,
                                hop: -1,
                            },
                            "6": {
                                gid: "0",
                                parentDbIndex: 0,
                                hop: -1,
                            },
                        },
                        public: {},
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
                        } as View,
                    ],
                } as Partial<SingleDashboard>,
            ],
        },
    },
});

describe("test sparse feature header", () => {
    it("test render", () => {
        const myStore = useMyStore();
        expect(
            myStore.singleDashboardList[0].viewsDefinitionList[0].diffColorRange
        ).toBeUndefined();
        const wrapper = mount(SparseFeatureHeader, {
            props: props,
            global: {
                plugins: [testPinia, ElementPlus],
            },
        });

        expect(
            myStore.singleDashboardList[0].viewsDefinitionList[0].diffColorRange
        ).toEqual([0, 1]);

        wrapper.unmount();
    });
});
