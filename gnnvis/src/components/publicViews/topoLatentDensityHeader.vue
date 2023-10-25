<template>
    <el-tooltip effect="dark" placement="bottom">
        <template #content>
            the domain extent of color scale, using local relative extent
            (min-max) or absolute data (in [0,1])
        </template>
        <el-switch
            v-model="view.isRelativeColorScale"
            inactive-text="0-1 (absolute)"
            active-text="min-max (relative)"
        ></el-switch>
    </el-tooltip>

    <el-tooltip effect="light" placement="top">
        <template #content>
            <div v-if="isDbWiseComparativeDb(db)">
                Grey level encodes distance between two nodes selections(with
                neighbors) from two dashboards.
                <br />
                For example, sel_1: {a, b, c}, sel_2: {c, d, e}. For 'a', we
                calc dist(a,d) and dist(a,e) and their avg, and 'c' was exempted
                from calculation. As with other cases.
            </div>
            <div v-else>
                Grey level encodes distance between nodes selection(with
                neighbors) and the other nodes in current dashboards.
                <br />
                For example, sel_1: {a, b}, dashboardFullDict: {a, b, c, d, e}.
                For 'a', we calc dist(a,c) and dist(a,d) and dist(a,e), and then
                calc avg. As with other cases.
            </div>
        </template>
        <el-icon style="height: 1.5em; width: 1.5em"><InfoFilled /></el-icon>
    </el-tooltip>

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
            <h3>Topo - Latent Density View Settings</h3>
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
import { InfoFilled } from "@element-plus/icons-vue";
import { useMyStore } from "@/stores/store";
import type {
    SingleDashboard,
    CompDashboard,
    LinkableView,
    NodeView,
    Type_NodeId,
    NodeCoord,
} from "@/types/types";
import { default as setting } from "@/components/icon/FluentSettings48Regular.vue";
import { nodeMapGraph2GraphMapNodes } from "@/utils/graphUtils";
import { isDbWiseComparativeDb } from "@/types/typeFuncs";
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
    which: {
        type: Number,
        default: 1,
        required: true,
    },
});
defineOptions({
    //因为我们没有用一个整体的div包起来，需要禁止透传
    inheritAttrs: false, //NOTE vue 3.3+
});
const myStore = useMyStore();
const db = myStore.getTypeReducedDashboardById(props.dbId)!;
const thisDs =
    props.which == 0
        ? myStore.getDatasetByName((db as SingleDashboard).refDatasetName)
        : props.which == 1
        ? myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[0])
        : myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[1]);
const theOtherDs =
    props.which == 0
        ? undefined
        : props.which == 1
        ? myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[1])
        : myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[0]);
const view = myStore.getViewByName(db, props.viewName) as NodeView<
    Node & NodeCoord
> &
    LinkableView;

//NOTE view的这些属性赋值，放在这里而不是initialNewView或者dashboard-view.setAttr，也能起到提前初始化的作用
view.isShowLinks = true;
view.linkOpacity = 0.8;
view.nodeRadius = 2;
view.isShowHopSymbols = true;
view.isRelativeColorScale = true;
</script>

<style scoped></style>
