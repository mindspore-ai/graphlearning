<template>
    <el-tooltip effect="dark" placement="top">
        <template #content> drag to swap/sort </template>

        <span class="handle">
            <handleSvg />
            {{ " " }}
            {{ props.viewName }}
        </span>
    </el-tooltip>
    <div class="brush&pan-buttons">
        <el-tooltip effect="dark" placement="top">
            <template #content>reset zoom</template>
            <el-button @click="(e:Event) => handleResetZoomClick(e)">
                <resetZoomSvg />
            </el-button>
        </el-tooltip>
        <el-divider direction="vertical" />
        <el-tooltip effect="dark" placement="top">
            <template #content>
                enable pan and disable brush<br />but don't clear selection<br />instead
                use "clear selection" if need clear
            </template>
            <el-button
                @click="(e:Event) => handlePanClick(e)"
                :type="view.isBrushEnabled ? '' : 'primary'"
            >
                <panSvg />
            </el-button>
        </el-tooltip>
        <el-divider direction="vertical" />
        <el-tooltip
            effect="dark"
            content="enable brush and disable pan"
            placement="top"
        >
            <el-button
                @click="(e:Event) => handleBrushClick(e)"
                :type="view.isBrushEnabled ? 'primary' : ''"
            >
                <brushSvg />
            </el-button>
        </el-tooltip>
    </div>
</template>

<script setup lang="ts">
// import { useUIStore } from "../../stores/ui";
import { default as brushSvg } from "../icon/PhSelectionPlusDuotone.vue";
import { default as panSvg } from "../icon/MaterialSymbolsPanToolOutline.vue";
import { default as resetZoomSvg } from "../icon/CarbonCenterToFit.vue";
import { useMyStore } from "@/stores/store";

import { default as handleSvg } from "../icon/RadixIconsDragHandleHorizontal.vue";
const props = defineProps({
    dbId: {
        type: String,
        default: "",
        required: true,
    },
    viewName: {
        type: String,
        default: "Graph",
    },
});
const myStore = useMyStore();
const view = myStore.getViewByName(props.dbId, props.viewName)!;
// 按钮
const handleResetZoomClick = (e: Event) => {
    console.log(`in ${props.viewName},  resetZoom clicked`);
    view.resetZoomFunc();
    myStore.repairButtonFocus(e);
};
const handlePanClick = async (e: Event) => {
    console.log(`in ${props.viewName}, pan clicked, view is`, view);

    view.isBrushEnabled = false;
    view.brushDisableFunc();
    view.panEnableFunc();
    myStore.repairButtonFocus(e);
};
const handleBrushClick = async (e: Event) => {
    console.log(`in ${props.viewName},  brush clicked`);
    view.isBrushEnabled = true;
    view.brushEnableFunc();
    view.panDisableFunc();
    myStore.repairButtonFocus(e);
};
</script>

<style scoped>
.handle {
    cursor: move;
}
</style>
