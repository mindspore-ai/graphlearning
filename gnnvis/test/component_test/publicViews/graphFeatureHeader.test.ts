import { createTestingPinia } from "@pinia/testing";
import * as d3 from "d3";
import ElementPlus from "element-plus";
import "element-plus/dist/index.css";
import { server } from "../../mocks/server";
import {
    describe,
    it,
    expect,
    beforeAll,
    afterAll,
    beforeEach,
    afterEach,
} from "vitest";

import { mount } from "@vue/test-utils";
import fetch from "node-fetch";

import GraphFeatureRenderer from "../../../src/components/publicViews/graphFeatureRenderer.vue";
import GraphFeatureHeader from "../../../src/components/publicViews/graphFeatureHeader.vue";
import { Dataset, SingleDashboard, View } from "../../../src/types/types";
import { useMyStore } from "../../../src/stores/store";
import { isEmptyDict } from "../../../src/utils/graphUtils";

const props: InstanceType<typeof GraphFeatureRenderer>["$props"] = {
    dbId: "db-test",
    viewName: "Graph Feature Space",
};

const testPinia = createTestingPinia({
    // createSpy: vi.fn,
    stubActions: false,
    initialState: {
        my: {
            // ...myStore,

            datasetList: [
                {
                    name: "test-graph-feature-dataset",
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
                    refDatasetName: "test-graph-feature-dataset",
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
                        } as View,
                    ],
                } as Partial<SingleDashboard>,
            ],
        },
    },
});

describe("test graph feature renderer", () => {
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

    it("test click fetch render", async () => {
        const myStore = useMyStore();
        const wrapper = mount(
            {
                template: `<div>
                <GraphFeatureHeader
                    :dbId="dbId" :viewName="viewName"  />
                <GraphFeatureRenderer
                    :dbId="dbId" :viewName="viewName"  />
                </div>`,

                //以下两种方式皆可
                //父的props再传子的props
                // props: {
                //     dbId: String,
                //     viewName: String,
                //     which: Number,
                // },

                //父的data传子的props
                data() {
                    return {
                        ...props,
                    };
                },
                components: { GraphFeatureRenderer, GraphFeatureHeader },
            },
            {
                global: {
                    plugins: [testPinia, ElementPlus],
                    components: { GraphFeatureRenderer, GraphFeatureHeader },
                },
                // props:props
            }
        );

        const buttons = wrapper.findAll("button");
        expect(buttons).toHaveLength(2);
        await buttons[0].trigger("click");
        await new Promise<void>((resolve, reject) => {
            setTimeout(() => {
                resolve();
            }, 2000);
        });

        console.log(myStore.datasetList[0].graphUserDefinedFeatureRecord);

        expect(
            isEmptyDict(myStore.datasetList[0].graphUserDefinedFeatureRecord)
        ).toBeDefined();
        expect(
            isEmptyDict(myStore.datasetList[0].graphUserDefinedFeatureRecord)
        ).toBe(false);
        expect(wrapper.findAll(".block")).toHaveLength(3);

        wrapper.unmount();
    });
});
