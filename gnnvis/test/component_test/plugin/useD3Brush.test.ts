import { createTestingPinia } from "@pinia/testing";
import { createPinia } from "pinia";
import { useMyStore } from "../../../src/stores/store";

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

import GraphRenderer from "../../../src/components/publicViews/graphRenderer.vue";
import { useD3Brush } from "../../../src/components/plugin/useD3Brush";
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
                } as Partial<Dataset>,
            ],
            singleDashboardList: [
                {
                    id: "db-test",
                    name: "dbName",
                    refDatasetName: "dataset-test",
                    isRoot: true,
                    labelType: "true",
                    clearSelMode: "manual",
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
                            isShowLinks: true,
                            linkOpacity: 0.8,
                            nodeRadius: 2,
                            isShowHopSymbols: false,
                        } as NodeView<
                            Node & NodeCoord & d3.SimulationNodeDatum
                        > &
                            LinkableView,
                    ],
                } as Partial<SingleDashboard>,
            ],
        },
    },
});

// const dom = new JSDOM(`<!DOCTYPE html><div id='app'></div>`);
window.document.body.innerHTML = `<div>
    <div id="app"></div>
  </div>`;
// window = dom.window;
// document = window.document;
describe("test useD3Brush", () => {
    it("test brush in topology space", async () => {
        const wrapper = mount(GraphRenderer, {
            props: props,
            global: {
                plugins: [testPinia],
            },
            attachTo: document.getElementById("app"),
        });
        const gBrush = wrapper.find(".gBrush");
        expect(gBrush.html()).toContain("rect");

        const rectangle = gBrush.find(".overlay");
        //先尝试了模拟事件的target，不让，然后看看这个事件在真实环境到底有什么不一样
        // 然后尝试在test环境打印这个事件，那就要额外绑定listener,
        // rectangle.element.addEventListener("mousedown", (e) => {
        //     console.log("in test ,mousedown ,e is", e.target);
        // });
        console.log(document.body.innerHTML);
        console.log(wrapper.html());
        const myStore = useMyStore();
        const {
            bodyHeight,
            bodyWidth,
            bodyMargins: { top, bottom, left, right },
        } = myStore.singleDashboardList[0].viewsDefinitionList[0];

        console.log(gBrush.element.__brush);
        await rectangle.trigger("mousedown", {
            x: bodyWidth * (left / 2),
            y: bodyHeight * (top / 2),
            offsetX: bodyWidth * (left / 2),
            offsetY: bodyHeight * (top / 2),
            clientX: bodyWidth * (left / 2),
            clientY: bodyHeight * (top / 2),
            pageX: bodyWidth * (left / 2),
            pageY: bodyHeight * (top / 2),
            layerX: bodyWidth * (left / 2),
            layerY: bodyHeight * (top / 2),
            screenX: bodyWidth * (left / 2),
            screenY: bodyHeight * (top / 2),
            // isTrusted: true,
            // bubbles: true,
            // button: 0,
            // buttons: 1,
            // cancelable: true,

            view: window,
        });
        await rectangle.trigger("mousemove", {
            x: bodyWidth * (1 - right / 2),
            y: bodyHeight * (1 - bottom / 2),
            offsetX: bodyWidth * (1 - right / 2),
            offsetY: bodyHeight * (1 - bottom / 2),
            clientX: bodyWidth * (1 - right / 2),
            clientY: bodyHeight * (1 - bottom / 2),
            view: window,
        });

        await rectangle.trigger("mouseup", {
            x: bodyWidth * (1 - right / 2),
            y: bodyHeight * (1 - bottom / 2),
            offsetX: bodyWidth * (1 - right / 2),
            offsetY: bodyHeight * (1 - bottom / 2),
            clientX: bodyWidth * (1 - right / 2),
            clientY: bodyHeight * (1 - bottom / 2),
            view: window,
        });

        const points = wrapper.findAll("circle");
        console.log(points.map((d) => d.html()));
        // points.forEach((w) => {
        //     expect(w.attributes("stroke")).toBe("black");
        // });
        expect(points).toHaveLength(7);

        wrapper.unmount();
    });
});
