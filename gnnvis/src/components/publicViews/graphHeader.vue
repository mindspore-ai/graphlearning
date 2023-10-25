<template>
    <el-popover
        placement="top-end"
        :trigger="myStore.settingsMenuTriggerMode"
        :width="400"
    >
        <template #reference>
            <setting
                style="
                    height: 1.5em;
                    width: 1.5em;
                    color: var(--el-text-color-regular);
                "
            />
        </template>

        <div style="text-align: start">
            <h3>Graph View Settings</h3>
            <el-divider />

            <span class="setting-item-name"> if show links:{{ " " }} </span>
            <el-switch
                v-model="view.isShowLinks"
                inactive-text="hide links"
                active-text="show links"
            ></el-switch>
            <br />

            <span class="setting-item-name"> link opacity:{{ " " }} </span>
            <el-slider
                :disabled="!view.isShowLinks"
                :min="0"
                :max="1"
                :step="0.01"
                v-model="view.linkOpacity"
            ></el-slider>
            <br />

            <span class="setting-item-name">node radius {{ " " }} </span>
            <el-slider
                v-model="view.nodeRadius"
                :min="2"
                :step="0.1"
                :max="
                    Math.max(Math.min(view.bodyWidth, view.bodyHeight) / 40, 10)
                "
            />
            <br />

            <span class="setting-item-name">show hop symbols {{ " " }} </span>

            <el-tooltip effect="dark" placement="bottom">
                <template #content>
                    whether to display hop neighbors using different
                    symbols/glyphs
                    <br />only available in non-root dashboard
                </template>
                <el-switch
                    :disabled="db.isRoot"
                    v-model="view.isShowHopSymbols"
                    inactive-text="circle"
                    active-text="different symbols"
                ></el-switch>
            </el-tooltip>
            <br />
        </div>
    </el-popover>
</template>

<script setup lang="ts">
import { toRaw, shallowRef, ref, watch, computed } from "vue";
import { useMyStore } from "@/stores/store";
import type {
    SingleDashboard,
    CompDashboard,
    LinkableView,
    NodeView,
    Type_NodeId,
} from "@/types/types";
import { default as setting } from "@/components/icon/FluentSettings48Regular.vue";
import { nodeMapGraph2GraphMapNodes } from "@/utils/graphUtils";
const props = defineProps({
    dbId: {
        type: String,
        default: "",
        required: true,
    },
    viewName: {
        type: String,
        required: true,
    },
});
const myStore = useMyStore();
const db = myStore.getTypeReducedDashboardById(props.dbId)!;
const thisDs = Object.hasOwn(db, "refDatasetsNames")
    ? myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[0])
    : myStore.getDatasetByName((db as SingleDashboard).refDatasetName);
const theOtherDs = Object.hasOwn(db, "refDatasetsNames")
    ? myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[1])
    : undefined;
const view = myStore.getViewByName(db, props.viewName) as NodeView<{
    id: Type_NodeId;
    x: number;
    y: number;
}> &
    LinkableView;

//NOTE view的这些属性赋值，放在这里而不是initialNewView或者dashboard-view.setAttr，也能起到提前初始化的作用
view.isShowLinks = true;
view.linkOpacity = 0.8;
view.nodeRadius = 2;
view.isShowHopSymbols = db.isRoot ? false : true;
</script>

<style scoped></style>
