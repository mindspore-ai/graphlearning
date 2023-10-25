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
import { useD3Zoom } from "../../../src/components/plugin/useD3Zoom";
import { Window, HTMLElement, GlobalWindow } from "happy-dom";
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
                            isBrushEnabled: false,
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
// const window = new Window({
//     innerWidth: 1024,
//     innerHeight: 768,
//     url: "http://localhost:6000",
// });
window.document.body.innerHTML = `<div>
    <div id="app"></div>
  </div>`;
// window = dom.window;
// document = window.document;
describe("test useD3Zoom", () => {
    it("test zoom in topology space", async () => {
        const wrapper = mount(GraphRenderer, {
            props: props,
            global: {
                plugins: [testPinia],
            },
            attachTo: document.getElementById("app"),
        });
        console.log(wrapper.html());
        const gTrans = wrapper.find("#globalTransform");
        const transformOld = gTrans.attributes("transform");
        const customEvent = new WheelEvent("wheel", {
            bubbles: true, // 模拟冒泡事件
            cancelable: true, // 可以被取消
            deltaY: -200, // 向上滚动
        });
        // 手动设置currentTarget属性

        // 触发自定义事件
        // wrapper.element.dispatchEvent(customEvent);
        // await new Promise<void>((resolve, reject) => {
        //     setTimeout(() => {
        //         resolve();
        //     }, 1000);
        // });
        await wrapper.trigger("wheel", {
            deltaY: -200, // 向上滚动
            isTrusted: true,
            altKey: false,
            bubbles: true,
            button: 0,
            buttons: 0,
            // cancelBubble: false,
            cancelable: true,
            clientX: 430,
            clientY: 429,
            composed: true,
            ctrlKey: false,
            // currentTarget: null,
            defaultPrevented: true,
            deltaMode: 0,
            deltaX: -0,
            deltaZ: 0,
            detail: 0,
            eventPhase: 0,
            fromElement: null,
            layerX: 430,
            layerY: 387,
            metaKey: false,
            movementX: 0,
            movementY: 0,
            offsetX: 388,
            offsetY: 239,
            pageX: 430,
            pageY: 429,
            relatedTarget: null,
            returnValue: false,
            screenX: 430,
            screenY: 506,
            shiftKey: false,
            sourceCapabilities: null,
            type: "wheel",
            // view: window,
            wheelDelta: 240,
            wheelDeltaX: 0,
            wheelDeltaY: 240,
            which: 0,
            x: 430,
            y: 429,
        });
        // await wrapper.trigger("mousedown", {
        //     x: 100,
        //     y: 100,
        //     view: window,
        // });
        // await wrapper.trigger("mousemove", {
        //     x: 200,
        //     y: 200,
        //     view: window,
        // });
        // await wrapper.trigger("mouseup", {
        //     view: window,
        // });
        const transformNew = gTrans.attributes("transform");
        console.log(transformOld, transformNew);
        wrapper.unmount();
    });
});
