<template>
    <el-popover
        placement="top-end"
        :trigger="myStore.settingsMenuTriggerMode"
        :width="500"
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
            <h3>Dense Feature View Settings</h3>
            <el-divider />

            <template v-if="!isDbWiseComparativeDb(db)">
                <span class="setting-item-name">columns:{{ " " }} </span>
                <el-slider
                    :min="1"
                    :max="5"
                    :step="1"
                    show-stops
                    v-model="viewDef.numColumns"
                ></el-slider>
                <br />
            </template>
            <template v-else>
                <span class="setting-item-name"
                    >columns: (should be an even number in dashboard-wise
                    comparison)</span
                >
                <el-slider
                    :min="2"
                    :max="6"
                    :step="2"
                    show-stops
                    v-model="viewDef.numColumns"
                ></el-slider>
                <br />
            </template>

            <span class="setting-item-name">sub view height:{{ " " }} </span>
            <el-tooltip
                effect="dark"
                placement="bottom"
                content="the height of each sub view, calculated by 0.4-0.9 view height"
            >
                <el-slider
                    :min="viewDef.bodyHeight * 0.4"
                    :max="viewDef.bodyHeight * 0.9"
                    :step="1"
                    v-model="viewDef.subHeight"
                ></el-slider>
            </el-tooltip>
            <br />

            <span class="setting-item-name"> relative/absolute:{{ " " }} </span>
            <el-tooltip
                effect="dark"
                placement="top"
                content="whether to show the relative value of histogram"
            >
                <el-switch
                    v-model="viewDef.isRelative"
                    inactive-text="absolute"
                    active-text="relative"
                    :loading="isRelativeSwitcherLoading"
                    :before-change="beforeIsRelativeChange"
                ></el-switch>
            </el-tooltip>
            <br />

            <span class="setting-item-name">
                correspondingly align same label :{{ " " }}
            </span>
            <el-tooltip
                effect="dark"
                placement="top"
                :content="`if on, when click a label to stack it to bottom,\n
                other sub histograms will align same labels to bottom like wise.`"
            >
                <el-switch
                    v-model="viewDef.isCorrespondAlign"
                    inactive-text="false"
                    active-text="true"
                ></el-switch>
            </el-tooltip>
            <br />
        </div>
    </el-popover>
</template>

<script setup lang="ts">
import { toRaw, shallowRef, ref, watch, computed } from "vue";
import { ElMessage } from "element-plus";
import { useMyStore } from "@/stores/store";
import type { SingleDashboard, DenseView, CompDashboard } from "@/types/types";
import { default as setting } from "@/components/icon/FluentSettings48Regular.vue";
import { isDbWiseComparativeDb } from "@/types/typeFuncs";
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
const viewDef = myStore.getViewByName(props.dbId, props.viewName) as DenseView;

//NOTE 放在这里而不是initialNewView或者dashboard-view.setAttr，也能起到提前初始化的作用
viewDef.numColumns = isDbWiseComparativeDb(db)
    ? 2
    : ds.denseNodeFeatures?.columns.length <= 1
    ? 1
    : 2;
viewDef.subHeight = viewDef.bodyHeight * 0.4;
viewDef.isCorrespondAlign = true;

const isRelativeSwitcherLoading = ref<boolean>(false);
const beforeIsRelativeChange = () => {
    isRelativeSwitcherLoading.value = true;
    return new Promise<boolean>((resolve, reject) => {
        setTimeout(() => {
            isRelativeSwitcherLoading.value = false;
            resolve(false);
        }, 3000);
        ElMessage({
            showClose: true,
            message: "Developing...",
            type: "warning",
            onClose: () => {
                isRelativeSwitcherLoading.value = false;
                resolve(false);
            },
        });
    });
};
</script>

<style scoped></style>
