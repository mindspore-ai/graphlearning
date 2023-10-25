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
            <h3>Multi Graph View Settings</h3>
            <el-divider />

            <span class="setting-item-name"> if show links:{{ " " }} </span>
            <el-switch
                v-model="viewDef.isShowLinks"
                inactive-text="hide links"
                active-text="show links"
            ></el-switch>
            <br />

            <span class="setting-item-name"> link opacity:{{ " " }} </span>
            <el-slider
                :disabled="!viewDef.isShowLinks"
                :min="0"
                :max="1"
                :step="0.01"
                v-model="viewDef.linkOpacity"
            ></el-slider>
            <br />

            <span class="setting-item-name">node radius {{ " " }} </span>
            <el-slider
                v-model="viewDef.nodeRadius"
                :min="2"
                :step="0.1"
                :max="
                    Math.max(
                        Math.min(viewDef.bodyWidth, viewDef.bodyHeight) / 40,
                        10
                    )
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
                    v-model="viewDef.isShowHopSymbols"
                    inactive-text="circle"
                    active-text="different symbols"
                ></el-switch>
            </el-tooltip>
            <br />

            <span class="setting-item-name">columns:{{ " " }} </span>
            <el-slider
                :min="2"
                :max="8"
                :step="1"
                show-stops
                v-model="viewDef.numColumns"
            ></el-slider>
            <br />

            <span class="setting-item-name">sub view height:{{ " " }} </span>
            <el-tooltip
                effect="dark"
                placement="bottom"
                content="the height of each sub view, calculated by 0.2-0.9 view height"
            >
                <el-slider
                    :min="viewDef.bodyHeight * 0.2"
                    :max="viewDef.bodyHeight * 0.9"
                    :disabled="viewDef.isAlignHeightAndWidth"
                    :step="1"
                    v-model="viewDef.subHeight"
                ></el-slider>
            </el-tooltip>
            <br />

            <span class="setting-item-name">
                align width and height :{{ " " }}
            </span>
            <el-tooltip
                effect="dark"
                placement="top"
                content="whether height = width for each view"
            >
                <el-switch v-model="viewDef.isAlignHeightAndWidth"></el-switch>
            </el-tooltip>
            <br />
        </div>
    </el-popover>
</template>

<script setup lang="ts">
import { toRaw, shallowRef, ref, watch, computed } from "vue";
import { ElMessage } from "element-plus";
import { useMyStore } from "@/stores/store";
import type {
    SingleDashboard,
    DenseView,
    CompDashboard,
    MultiGraphView,
} from "@/types/types";
import { default as setting } from "@/components/icon/FluentSettings48Regular.vue";
import { nodeMapGraph2GraphMapNodes } from "@/utils/graphUtils";
defineOptions({
    //因为我们没有用一个整体的div包起来，需要禁止透传
    inheritAttrs: false, //NOTE vue 3.3+
});
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
const ds = Object.hasOwn(db, "refDatasetsNames")
    ? myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[0]) ||
      myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[1])
    : myStore.getDatasetByName((db as SingleDashboard).refDatasetName);
const viewDef = myStore.getViewByName(
    props.dbId,
    props.viewName
) as MultiGraphView;

const numGraphs = Object.keys(
    nodeMapGraph2GraphMapNodes(db.srcNodesDict)
).length;
const initialNumColumns = 5;

//NOTE view的这些属性赋值，放在这里而不是initialNewView或者dashboard-view.setAttr，也能起到提前初始化的作用
viewDef.isShowLinks = true;
viewDef.linkOpacity = 0.8;
viewDef.numColumns =
    numGraphs <= initialNumColumns ? numGraphs : initialNumColumns;
viewDef.subHeight = viewDef.bodyHeight * 0.4;
viewDef.isAlignHeightAndWidth = true;
viewDef.nodeRadius = 2;
viewDef.isShowHopSymbols = db.isRoot ? false : true;
</script>

<style scoped></style>
