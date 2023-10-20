import { createTestingPinia } from "@pinia/testing";
import { useMyStore } from "../../../src/stores/store";
import { unref } from "vue";
import { Window, HTMLElement, GlobalWindow } from "happy-dom";
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

import { mount } from "@vue/test-utils";

import ResizableBox from "../../../src/components/publicViews/resizableBox.vue";
import type { Dataset, View, SingleDashboard } from "../../../src/types/types";

const props: InstanceType<typeof ResizableBox>["$props"] = {
    dbId: "db-test",
    viewName: "Topology Space",
};

const testPinia = createTestingPinia({
    createSpy: vi.fn,
    stubActions: false,
    initialState: {
        my: {
            // ...myStore,

            datasetList: [
                {
                    name: "dataset-test",
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
                            brushEnableFunc: vi.fn(),
                            //     () => { console.log("brush enabled");
                            // },
                            brushDisableFunc: vi.fn(),
                            //  () => {
                            //     console.log("brush disabled");
                            // },
                            panEnableFunc: vi.fn(),
                            //  () => {
                            //     console.log("brush enabled");
                            // },
                            panDisableFunc: vi.fn(),
                            //  () => {
                            //     console.log("pan disabled");
                            // },
                            hideRectWhenClearSelFunc: () => {
                                console.log("pan enabled");
                            },
                            resetZoomFunc: () => {
                                console.log("reset zoom");
                            },
                            isGettingSnapshot: false,
                            gettingSnapShotError: undefined,
                            snapshotBase64: "",

                            bodyProps: {},
                            headerProps: {},
                        } as View,
                    ],
                } as Partial<SingleDashboard>,
            ],
        },
    },
});

describe("test resizable box", () => {
    it("should ", async () => {
        const myStore = useMyStore();
        const wrapper = mount(
            {
                template: `<div>
                    <ResizableBox ref='child' :dbId="dbId" :viewName="viewName" />
                </div>`,
                props: {
                    dbId: String,
                    viewName: String,
                },
                components: { ResizableBox },
            },
            {
                props: props,
                global: {
                    components: { ResizableBox },
                    plugins: [testPinia],
                },
            }
        );
        console.log(Object.keys(wrapper.vm.$refs));
        console.log(wrapper.vm.$refs.child);

        expect(unref(wrapper.vm.$refs.child.widthStr)).toBe("500px");

        // not work!
        // const box = wrapper.find(".resize");
        // const resizeObserver = new ResizeObserver((entries) => {
        //     for (const entry of entries) {
        //         if (entry.contentBoxSize) {
        //             const contentBoxSize = entry.contentBoxSize[0];
        //             console.log(contentBoxSize.inlineSize);
        //         } else {
        //             console.log(entry.contentRect.width);
        //         }
        //     }
        //     console.log("Size changed");
        // });
        // resizeObserver.observe(box.element);
        // // how to trigger??
        wrapper.unmount();
    });
});
