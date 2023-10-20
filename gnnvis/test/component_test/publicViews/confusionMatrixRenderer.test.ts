import { createTestingPinia } from "@pinia/testing";

import * as d3 from "d3";

import { describe, it, expect } from "vitest";

import { mount } from "@vue/test-utils";

import ConfusionMatrixRenderer from "../../../src/components/publicViews/confusionMatrixRenderer.vue";
import { Dataset, SingleDashboard, View } from "../../../src/types/types";

const props: InstanceType<typeof ConfusionMatrixRenderer>["$props"] = {
    dbId: "db-test",
    viewName: "Prediction Space",
    which: 0,
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
                } as Partial<Dataset>,
            ],
            singleDashboardList: [
                {
                    id: "db-test",
                    refDatasetName: "dataset-test",
                    isRoot: true,
                    labelType: "true",
                    isHighlightCorrespondingNode: true,
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

describe("test confusion matrix renderer", () => {
    it("test render", () => {
        const wrapper = mount(ConfusionMatrixRenderer, {
            props: props,
            global: {
                plugins: [testPinia],
            },
        });
        const cells = wrapper.findAll(".cell");

        // predLabels: [1, 2, 3, 0, 3, 2, 1],
        // trueLabels: [0, 2, 1, 3, 0, 2, 2],
        expect(cells).toHaveLength(4 * 4);
        expect(cells[10].find("text").html()).toContain("2"); //2预测成2的有两个
        wrapper.unmount();
    });
});
