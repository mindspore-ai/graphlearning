import { createTestingPinia } from "@pinia/testing";

import { describe, it, expect } from "vitest";

import { Window } from "happy-dom";
import { mount } from "@vue/test-utils";

import LinkPredRenderer from "../../../src/components/publicViews/linkPredRenderer.vue";
import {
    Dataset,
    LinkPredView,
    SingleDashboard,
    View,
} from "../../../src/types/types";
import { useMyStore } from "../../../src/stores/store";

const props: InstanceType<typeof LinkPredRenderer>["$props"] = {
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
                    taskType: "link-prediction",
                    trueAllowEdges: [
                        ["0", "1"],
                        ["1", "2"],
                        ["2", "3"],
                        ["3", "4"],
                    ],
                    falseAllowEdges: [
                        ["4", "5"],
                        ["5", "6"],
                    ],
                    trueUnseenTopK: 3,
                    trueUnseenEdgesSorted: {
                        "0": ["3", "4", "5"],
                        "1": ["2", "3", "4"],
                        "2": ["4", "5", "6"],
                        "3": ["0", "1", "5"],
                        "4": ["1", "2", "6"],
                        "5": ["3", "0", "2"],
                        "6": ["2", "1", "0"],
                    },
                    globalNodesDict: {
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
                    hops: 2,
                    neighborMasks: ["7", "f", "1e", "3c", "78", "70", "60"],
                    neighborMasksByHop: [
                        ["3", "7", "e", "1c", "38", "70", "60"],
                        ["6", "e", "1c", "38", "70", "60", "40"],
                    ],
                    neighborMasksByHopPure: [
                        ["3", "7", "e", "1c", "38", "70", "60"],
                        ["4", "9", "12", "24", "48", "10", "20"],
                    ],
                } as Partial<Dataset>,
            ],
            singleDashboardList: [
                {
                    id: "db-test",
                    refDatasetName: "dataset-test",
                    isRoot: true,
                    labelType: "true",
                    isHighlightCorrespondingNode: true,
                    srcNodesDict: {
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
                    graphCoordsRet: [
                        { id: "0", x: 10, y: 10 },
                        { id: "1", x: 20, y: 20 },
                        { id: "2", x: 30, y: 30 },
                        { id: "3", x: 40, y: 40 },
                        { id: "4", x: 50, y: 50 },
                        { id: "5", x: 60, y: 60 },
                        { id: "6", x: 70, y: 70 },
                    ],
                    srcLinksArr: [
                        { eid: "0", source: "0", target: "1", gid: "0" },
                        { eid: "1", source: "1", target: "2", gid: "0" },
                        { eid: "2", source: "2", target: "3", gid: "0" },
                        { eid: "3", source: "3", target: "4", gid: "0" },
                        { eid: "4", source: "4", target: "5", gid: "0" },
                        { eid: "5", source: "5", target: "6", gid: "0" },
                    ],
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

                            currentHops: 1,
                            isShowTrueUnseen: false,
                            setAttr(attr, value: any) {
                                this[attr] = value;
                                return this;
                            },
                        } as LinkPredView,
                    ],
                } as Partial<SingleDashboard>,
            ],
        },
    },
});
describe("test link pred renderer", () => {
    it("test render", async () => {
        const myStore = useMyStore();
        const wrapper = mount(LinkPredRenderer, {
            props: props,
            global: {
                plugins: [testPinia],
            },
        });

        const window = new Window({
            innerWidth: 1024,
            innerHeight: 768,
            url: "http://localhost:6000",
        });
        // myStore.singleDashboardList[0].nodesSelections["public"] = {
        //     "0": {},
        //     "1": {},
        //     "2": {},
        //     "3": {},
        // };

        // await new Promise<void>((resolve, reject) => {
        //     setTimeout(() => {
        //         resolve();
        //     }, 800);
        // });

        wrapper.unmount();
    });
});
