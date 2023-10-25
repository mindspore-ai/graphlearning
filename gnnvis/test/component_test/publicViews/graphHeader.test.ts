import { createTestingPinia } from "@pinia/testing";
import { createPinia } from "pinia";
import { useMyStore } from "../../../src/stores/store";

import ElementPlus from "element-plus";
import "element-plus/dist/index.css";

import * as d3 from "d3";

import {
    describe,
    beforeEach,
    beforeAll,
    afterEach,
    afterAll,
    it,
    expect,
    vi,
} from "vitest";

// below two are mutually alternative
import { mount, shallowMount } from "@vue/test-utils";
import { render } from "@testing-library/vue";

import GraphRenderer from "../../../src/components/publicViews/graphRenderer.vue";
import GraphHeader from "../../../src/components/publicViews/graphHeader.vue";
import {
    CompDashboard,
    Dataset,
    LinkableView,
    NodeCoord,
    NodeView,
    SingleDashboard,
} from "../../../src/types/types";

const props: InstanceType<typeof GraphRenderer>["$props"] = {
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
                    name: "dataset-test",
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
                    isHighlightCorrespondingNode: true,
                    nodesSelections: {
                        full: {
                            "0": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "1": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "2": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "3": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "4": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "5": { gid: "0", parentDbIndex: 0, hop: -1 },
                            "6": { gid: "0", parentDbIndex: 0, hop: -1 },
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
                                console.log(
                                    "view definition of",
                                    this.viewName,
                                    ": now setting attr",
                                    attr,
                                    "before:",
                                    this[attr],
                                    "new",
                                    value
                                );
                                this[attr] = value;
                                return this;
                            },
                            // isShowLinks: true,
                            // linkOpacity: 0.8,
                            // nodeRadius: 2,
                            // isShowHopSymbols: true,
                        } as NodeView<
                            Node & NodeCoord & d3.SimulationNodeDatum
                        > &
                            LinkableView,
                    ],
                } as Partial<SingleDashboard>,
            ],
            getSingleDashboardById(id: string) {
                const ret = this.singleDashboardList.value.find(
                    (db) => db.id === id
                );
                console.log(
                    `search id ${id} in single db list!  got ret ${ret}`
                );
                if (ret) return ret;
            },
            getTypeReducedDashboardById(
                db: CompDashboard | SingleDashboard | string
            ): CompDashboard | SingleDashboard | undefined {
                if (typeof db === "string") {
                    let dbObj;

                    try {
                        console.log(
                            "ready to search single db list!, db is",
                            db
                        );
                        dbObj = this.getSingleDashboardById(db);
                        if (dbObj) return dbObj;
                    } catch (ee) {
                        throw new Error(
                            "Neither single or comparative dashboard found!, param: " +
                                db
                        );
                    }
                    // }
                } else {
                    return db;
                }
            },
        },
    },
});

it("test topology space with header", async () => {
    // console.log("before mount!!!!");
    // const localComponent = {
    //     template: `<div>
    //     <GraphHeader dbId=${props.dbId} viewName=${props.viewName}  />
    //     <GraphRenderer dbId=${props.dbId} viewName=${props.viewName} />
    //     </div>`,
    // };

    const myStore = useMyStore();
    expect(
        myStore.singleDashboardList[0].viewsDefinitionList[0].nodeRadius
    ).toBeUndefined();

    const wrapper = mount(GraphHeader, {
        props: props,
        global: {
            plugins: [testPinia, ElementPlus],
        },
    });

    //test onCreated should change radius
    expect(myStore.singleDashboardList[0].viewsDefinitionList).toBeDefined();
    expect(
        myStore.singleDashboardList[0].viewsDefinitionList[0].nodeRadius
    ).toBe(2);

    const settingSvg = wrapper.find(".el-tooltip__trigger");
    expect(settingSvg.exists()).toBe(true);

    // not work!!!
    // await settingSvg.trigger("click");
    // await new Promise<void>((resolve, reject) => {
    //     setTimeout(() => {
    //         resolve();
    //     }, 1500);
    // });
    // const popover = wrapper.find(".el-popper");
    // expect(popover.exists()).toBe(true);

    // const slider = wrapper.find(".el-slider__runway");
    // expect(slider.exists()).toBe(true);
    // const sliderRect = slider.element.getBoundingClientRect();
    // const sliderWidth = sliderRect.width;
    // const clickX = sliderRect.left + sliderWidth * 0.3;
    // await wrapper.trigger("click", { clientX: clickX });
    // expect(
    //     myStore.singleDashboardList[0].viewsDefinitionList[0].nodeRadius
    // ).not.toBe(2);

    wrapper.unmount();
});
