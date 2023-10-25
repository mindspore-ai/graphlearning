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
            <h3>Sparse Feature View Settings</h3>
            <el-divider />

            <span class="setting-item-name">
                color interpolate adaptive to zoom:{{ " " }}
            </span>
            <el-switch
                v-model="viewDef.isAdaptiveColorInterpolate"
                :loading="isAdaptiveColorSwitcherLoading"
                :before-change="beforeIsAdaptiveColorChange"
            ></el-switch>
            <br />

            <template v-if="isDbWiseComparativeDb(db)">
                <span class="setting-item-name"
                    >diff color interpolate range:{{ " " }}
                    {{ viewDef.diffColorRange }}
                </span>
                <el-slider
                    v-model="viewDef.diffColorRange"
                    :disabled="viewDef.isAdaptiveColorInterpolate"
                    range
                    :min="0"
                    :step="0.01"
                    :max="1"
                />
                <br />

                <span class="setting-item-name"
                    >sel from db0 color interpolate range:{{ " " }}
                    {{ viewDef.sel0ColorRange }}
                </span>
                <el-slider
                    v-model="viewDef.sel0ColorRange"
                    :disabled="viewDef.isAdaptiveColorInterpolate"
                    range
                    :min="0"
                    :step="0.01"
                    :max="1"
                />
                <br />

                <span class="setting-item-name"
                    >sel from db1 color interpolate range:{{ " " }}
                    {{ viewDef.sel1ColorRange }}
                </span>
                <el-slider
                    v-model="viewDef.sel1ColorRange"
                    :disabled="viewDef.isAdaptiveColorInterpolate"
                    range
                    :min="0"
                    :step="0.01"
                    :max="1"
                />
                <br />
            </template>
            <template v-else>
                <span class="setting-item-name"
                    >diff color interpolate range:{{ " " }}
                    {{ viewDef.diffColorRange }}
                </span>
                <el-slider
                    v-model="viewDef.diffColorRange"
                    :disabled="viewDef.isAdaptiveColorInterpolate"
                    range
                    :min="0"
                    :step="0.01"
                    :max="1"
                />
                <br />

                <span class="setting-item-name"
                    >sel color interpolate range:{{ " " }}
                    {{ viewDef.selColorRange }}
                </span>
                <el-slider
                    v-model="viewDef.selColorRange"
                    :disabled="viewDef.isAdaptiveColorInterpolate"
                    range
                    :min="0"
                    :step="0.01"
                    :max="1"
                />
                <br />
            </template>
        </div>
    </el-popover>
</template>

<script setup lang="ts">
import { toRaw, shallowRef, ref, watch, computed, onMounted } from "vue";
import { useMyStore } from "@/stores/store";
import type { SingleDashboard, DenseView, SparseView } from "@/types/types";
import { default as setting } from "@/components/icon/FluentSettings48Regular.vue";
import { ElMessage } from "element-plus";
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
const viewDef = myStore.getViewByName(props.dbId, props.viewName) as SparseView;
viewDef.isAdaptiveColorInterpolate = false;
if (isDbWiseComparativeDb(db)) {
    viewDef.diffColorRange = [0, 1]; //default
    viewDef.sel0ColorRange = [0.5, 1]; //default
    viewDef.sel1ColorRange = [0.5, 1]; //default
} else {
    viewDef.diffColorRange = [0, 1]; //default
    viewDef.selColorRange = [0.5, 1]; //default
}

const isAdaptiveColorSwitcherLoading = ref<boolean>(false);
const beforeIsAdaptiveColorChange = () => {
    isAdaptiveColorSwitcherLoading.value = true;
    return new Promise<boolean>((resolve, reject) => {
        setTimeout(() => {
            isAdaptiveColorSwitcherLoading.value = false;
            resolve(false);
        }, 3000);
        ElMessage({
            showClose: true,
            message: "Developing...",
            type: "warning",
            onClose: () => {
                isAdaptiveColorSwitcherLoading.value = false;
                resolve(false);
            },
        });
    });
};
</script>

<style scoped></style>
