import { describe, it, expect } from "vitest";
import { mount } from "@vue/test-utils";

import ScatterSymbolLegend from "../../../src/components/header/scatterSymbolLegend.vue";

describe("test scatterSymbolLegend.vue", () => {
    it("test render", () => {
        const wrapper = mount(ScatterSymbolLegend, {});
        const texts = wrapper.findAll("svg text");

        expect(texts).toHaveLength(8 + 1 + 3);
        // 总共12个：
        // "selection",
        // "unseen",
        // `hop=${1}`,
        // `hop=${2}`,
        // `hop=${3}`,
        // `hop=${4}`,
        // `hop=${5}`,
        // `hop>${5}`,
        // in dashboard-wise comparison:
        // 'from db0',
        // 'from db1',
        // 'from db0 and db1 in percents',

        wrapper.unmount();
    });
});
