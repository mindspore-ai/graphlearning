import { createTestingPinia } from "@pinia/testing";
import { useMyStore } from "../../../src/stores/store";

import ElementPlus from "element-plus";
import "element-plus/dist/index.css";

import * as d3 from "d3";

import { describe, it, expect } from "vitest";

import { mount } from "@vue/test-utils";

import TsneRenderer from "../../../src/components/publicViews/tsneRenderer.vue";
import TsneHeader from "../../../src/components/publicViews/tsneHeader.vue";
import {
    CompDashboard,
    Dataset,
    LinkableView,
    AggregatedView,
    SingleDashboard,
} from "../../../src/types/types";

const props: InstanceType<typeof TsneRenderer>["$props"] = {
    dbId: "db-test",
    viewName: "Latent Space",
    which: 0,
};
const testPinia = createTestingPinia({
    // createSpy: vi.fn,
    stubActions: false,
    initialState: {
        my: {
            datasetList: [
                {
                    name: "dataset-test",
                    taskType: "node-classification",
                    predLabels: [1, 2, 3, 0, 3, 2, 1],
                    trueLabels: [0, 2, 1, 3, 0, 2, 2],
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
                    tsneRet: [
                        { id: "0", x: 10, y: 10 },
                        { id: "1", x: 20, y: 20 },
                        { id: "2", x: 30, y: 30 },
                        { id: "3", x: 40, y: 40 },
                        { id: "4", x: 50, y: 50 },
                        { id: "5", x: 60, y: 60 },
                        { id: "6", x: 70, y: 70 },
                    ],
                    srcLinksArr: [
                        {
                            eid: "0",
                            source: "0",
                            target: "1",
                            gid: "0",
                        },
                        {
                            eid: "1",
                            source: "1",
                            target: "2",
                            gid: "0",
                        },
                        {
                            eid: "2",
                            source: "2",
                            target: "3",
                            gid: "0",
                        },
                        {
                            eid: "3",
                            source: "3",
                            target: "4",
                            gid: "0",
                        },
                        {
                            eid: "4",
                            source: "4",
                            target: "5",
                            gid: "0",
                        },
                        {
                            eid: "5",
                            source: "5",
                            target: "6",
                            gid: "0",
                        },
                    ],
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
                            isShowLinks: true,
                            linkOpacity: 0.8,
                            // nodeRadius: 2,
                            // isShowHopSymbols: false,
                            isShowAggregation: false,
                        } as AggregatedView & LinkableView,
                    ],
                } as Partial<SingleDashboard>,
            ],
        },
    },
});

describe("test tsne header", () => {
    it("should change the node radius when header is mounted", () => {
        const myStore = useMyStore();
        expect(
            myStore.singleDashboardList[0].viewsDefinitionList[0].nodeRadius
        ).toBeUndefined();

        const wrapper = mount(
            {
                template: `<div>
                <TsneHeader
                    :dbId="dbId" :viewName="viewName" :which="which" />
                <TsneRenderer
                    :dbId="dbId" :viewName="viewName" :which="which" />
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
                components: { TsneRenderer, TsneHeader },
            },
            {
                global: {
                    plugins: [testPinia, ElementPlus],
                    components: { TsneRenderer, TsneHeader },
                },
                // props:props
            }
        );

        expect(
            myStore.singleDashboardList[0].viewsDefinitionList
        ).toBeDefined();
        expect(
            myStore.singleDashboardList[0].viewsDefinitionList[0].nodeRadius
        ).toBe(2);

        const circles = wrapper.findAll("circle");
        expect(circles).toHaveLength(7);
        const lines = wrapper.findAll("line");
        expect(lines).toHaveLength(6);

        wrapper.unmount();
    });
});
