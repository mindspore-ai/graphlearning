import { createTestingPinia } from "@pinia/testing";
import { useMyStore } from "../../../src/stores/store";
import ElementPlus from "element-plus";
import "element-plus/dist/index.css";

import * as d3 from "d3";

import { describe, it, expect } from "vitest";

import { mount } from "@vue/test-utils";

import DenseFeatureHeader from "../../../src/components/publicViews/denseFeatureHeader.vue";
import { Dataset, SingleDashboard, View } from "../../../src/types/types";

const props: InstanceType<typeof DenseFeatureHeader>["$props"] = {
    dbId: "db-test",
    viewName: "Feature Space",
};
const denseNodeFeatures: d3.DSVParsedArray<{
    id: string;
    [key: string]: unknown;
}> = [
    { id: "0", feat0: 0.1, feat1: 0.2, feat2: 0.3 },
    { id: "1", feat0: 0.1, feat1: 0.2, feat2: 0.3 },
    { id: "2", feat0: 0.1, feat1: 0.2, feat2: 0.3 },
    { id: "3", feat0: 0.1, feat1: 0.2, feat2: 0.3 },
    { id: "4", feat0: 0.1, feat1: 0.2, feat2: 0.3 },
    { id: "5", feat0: 0.1, feat1: 0.2, feat2: 0.3 },
    { id: "6", feat0: 0.1, feat1: 0.2, feat2: 0.3 },
];

denseNodeFeatures.columns = ["feat0", "feat1", "feat2"];

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

                    denseNodeFeatures: denseNodeFeatures,
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

describe("test dense feature header", () => {
    it("test render", () => {
        const myStore = useMyStore();
        expect(
            myStore.singleDashboardList[0].viewsDefinitionList[0].numColumns
        ).toBeUndefined();
        const wrapper = mount(DenseFeatureHeader, {
            props: props,
            global: {
                plugins: [testPinia, ElementPlus],
            },
        });
        expect(
            myStore.singleDashboardList[0].viewsDefinitionList[0].numColumns
        ).toBe(2);

        wrapper.unmount();
    });
});
