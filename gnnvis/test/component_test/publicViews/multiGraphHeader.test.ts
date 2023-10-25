import { createTestingPinia } from "@pinia/testing";
import * as d3 from "d3";
import ElementPlus from "element-plus";
import "element-plus/dist/index.css";
import { describe, it, expect } from "vitest";

import { mount } from "@vue/test-utils";

import MultiGraphHeader from "../../../src/components/publicViews/multiGraphHeader.vue";
import {
    Dataset,
    MultiGraphView,
    SingleDashboard,
    View,
} from "../../../src/types/types";
import { useMyStore } from "../../../src/stores/store";

const props: InstanceType<typeof MultiGraphHeader>["$props"] = {
    dbId: "db-test",
    viewName: "Topology Space",
};

const testPinia = createTestingPinia({
    // createSpy: vi.fn,
    stubActions: false,
    initialState: {
        my: {
            // ...myStore,

            datasetList: [
                {
                    name: "test-multi-graph",
                    taskType: "graph-classification",
                    graphPredLabels: [1, 2, 3, 0, 3, 2, 1],
                    graphTrueLabels: [0, 2, 1, 3, 0, 2, 2],
                    graphArr: [
                        { gid: "0", nodes: ["0"], links: [] },
                        { gid: "1", nodes: ["1"], links: [] },
                        { gid: "2", nodes: ["2"], links: [] },
                        { gid: "3", nodes: ["3"], links: [] },
                        { gid: "4", nodes: ["4"], links: [] },
                        { gid: "5", nodes: ["5"], links: [] },
                        { gid: "6", nodes: ["6"], links: [] },
                    ],
                    numGraphClasses: 4,
                } as Partial<Dataset>,
            ],
            singleDashboardList: [
                {
                    id: "db-test",
                    refDatasetName: "test-multi-graph",
                    isRoot: true,
                    labelType: "true",
                    isHighlightCorrespondingNode: true,
                    parentId: "0",
                    srcLinksDict: {},
                    srcLinksArr: [],
                    graphCoordsRet: [
                        { id: "0", x: 10, y: 10 },
                        { id: "1", x: 20, y: 20 },
                        { id: "2", x: 30, y: 30 },
                        { id: "3", x: 40, y: 40 },
                        { id: "4", x: 50, y: 50 },
                        { id: "5", x: 60, y: 60 },
                        { id: "6", x: 70, y: 70 },
                    ],
                    srcNodesDict: {
                        "0": {
                            gid: "0",
                            parentDbIndex: 0,
                            hop: -1,
                        },
                        "1": {
                            gid: "1",
                            parentDbIndex: 0,
                            hop: -1,
                        },
                        "2": {
                            gid: "2",
                            parentDbIndex: 0,
                            hop: -1,
                        },
                        "3": {
                            gid: "3",
                            parentDbIndex: 0,
                            hop: -1,
                        },
                        "4": {
                            gid: "4",
                            parentDbIndex: 0,
                            hop: -1,
                        },
                        "5": {
                            gid: "5",
                            parentDbIndex: 0,
                            hop: -1,
                        },
                        "6": {
                            gid: "6",
                            parentDbIndex: 0,
                            hop: -1,
                        },
                    },
                    nodesSelections: {
                        full: {
                            "0": {
                                gid: "0",
                                parentDbIndex: 0,
                                hop: -1,
                            },
                            "1": {
                                gid: "1",
                                parentDbIndex: 0,
                                hop: -1,
                            },
                            "2": {
                                gid: "2",
                                parentDbIndex: 0,
                                hop: -1,
                            },
                            "3": {
                                gid: "3",
                                parentDbIndex: 0,
                                hop: -1,
                            },
                            "4": {
                                gid: "4",
                                parentDbIndex: 0,
                                hop: -1,
                            },
                            "5": {
                                gid: "5",
                                parentDbIndex: 0,
                                hop: -1,
                            },
                            "6": {
                                gid: "6",
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
                            // isShowLinks: true,
                            // linkOpacity: 0.8,
                            // numColumns: 3,
                            // isAlignHeightAndWidth: true,
                            // nodeRadius: 2,
                            // isShowHopSymbols: false,
                        } as MultiGraphView,
                    ],
                } as Partial<SingleDashboard>,
            ],
        },
    },
});

describe("test multi graph renderer", () => {
    it("test render", () => {
        const store = useMyStore();
        expect(
            store.singleDashboardList[0].viewsDefinitionList[0].numColumns
        ).toBeUndefined();
        const wrapper = mount(MultiGraphHeader, {
            props: props,
            global: {
                plugins: [testPinia, ElementPlus],
            },
        });
        expect(
            store.singleDashboardList[0].viewsDefinitionList[0].numColumns
        ).toBe(5);
        wrapper.unmount();
    });
});
