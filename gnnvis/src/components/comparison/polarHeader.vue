<template>
    <el-tooltip
        effect="dark"
        placement="top"
        content="select some nodes and calc polar first!"
        :disabled="db.polarWorker.workerStatus === 'SUCCESS'"
    >
        <el-switch
            :disabled="db.polarWorker.workerStatus !== 'SUCCESS'"
            v-model="view.isShowAggregation"
            inactive-text="origin points"
            active-text="aggregated points"
        ></el-switch>
    </el-tooltip>

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
            <h3>Comparative Polar View Settings</h3>
            <el-divider />

            <span class="setting-item-name"
                >use which predLabel:{{ " " }}
            </span>
            <el-switch
                v-model="view.isUsingSecondDbPredLabel"
                :disabled="
                    ds1.taskType !== 'graph-classification' ||
                    db.labelType === 'true'
                "
                inactive-text="db 0"
                active-text="db 1"
            >
            </el-switch>
            <br />

            <span class="setting-item-name"
                >aggregate hexbin radius:{{ " " }}
            </span>
            <el-slider
                :min="10"
                :max="100"
                :disabled="db.polarWorker.workerStatus !== 'SUCCESS'"
                v-model="view.meshRadius"
            ></el-slider>
            <br />

            <span class="setting-item-name"> if show mesh:{{ " " }} </span>
            <el-tooltip
                effect="dark"
                placement="top"
                :disabled="Boolean(view.hexbin)"
                content="run aggregation first to show mesh!"
            >
                <el-switch
                    :disabled="!Boolean(view.hexbin)"
                    v-model="view.isShowMesh"
                    inactive-text="hide mesh"
                    active-text="show mesh"
                ></el-switch>
            </el-tooltip>
            <br />

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
                :format-tooltip="(v:number)=>v/100"
                v-model="opacitySliderValue"
                @input="handleOpacitySliderChange"
            ></el-slider>
            <br />

            <span class="setting-item-name">topo dist algo:{{ " " }} </span>
            <el-tooltip effect="dark" placement="top">
                <template #content>
                    the algorithm of difference in topology space
                    <br />
                    shortedPath: simply calc the path len between the neighbor
                    and selected node(s)
                    <br />
                    jaccard: for each neighbor, calc the jaccard distance
                    between that neighbor and selected node(s)
                    <br />
                    hamming: for each neighbor, calc the hamming distance
                    between that neighbor and selected node(s)
                </template>
                <el-select
                    :style="{ width: 100 + 'px' }"
                    v-model="view.polarTopoDistAlgo"
                    placeholder="topo distance algorithm"
                    filterable
                >
                    <el-option
                        v-for="a in polarTopoDistAlgos"
                        :key="a"
                        :label="a"
                        :value="a"
                    />
                </el-select>
            </el-tooltip>
            <br />

            <span class="setting-item-name"> emb counting algo{{ " " }} </span>
            <el-tooltip effect="dark" placement="top">
                <template #content>
                    the algorithm of embs accounting
                    <br />
                    if single node selected, simply calc all distances from 'all
                    nodes' to neighbors of the selected node
                    <br />
                    center: treat nodes as a cluster, and calc all distances
                    from 'all nodes' to neighbors of the cluster
                    <br />
                    avg: for each node in 'all neighbor nodes', calc avg dist to
                    each neighbor of selected nodes, and repeat it for 'all
                    nodes'
                </template>
                <el-select
                    :style="{ width: 100 + 'px' }"
                    v-model="view.polarEmbDiffAlgo"
                    clearable
                    placeholder="select algorithm"
                >
                    <el-option
                        v-for="a in polarEmbDiffAlgos"
                        :key="a"
                        :label="a"
                        :value="a"
                    />
                </el-select>
            </el-tooltip>

            <el-divider direction="vertical" />

            <!-- NOTE 下面的v-model绑定nodesSelection['single']所以这里的值是单个字符串，而不是键值对了-->
            <el-tooltip
                effect="dark"
                placement="top"
                :disabled="view.polarEmbDiffAlgo === 'single'"
            >
                <template #content> only available in "single" algo </template>
                <el-select
                    :disabled="view.polarEmbDiffAlgo !== 'single'"
                    v-model="db.nodesSelections['comparativeSingle']"
                    filterable
                    clearable
                    placeholder="select one node"
                >
                    <el-option
                        v-for="n in Object.keys(
                            db.nodesSelections['comparative']
                        )"
                        :key="n"
                        :label="`id: ${n}, predLabel: ${
                            predLabels[+n]
                        } trueLabel: ${trueLabels[+n]}`"
                        :value="n"
                    />
                </el-select>
            </el-tooltip>
            <br />
        </div>
    </el-popover>
    <!-- 坑！不要在v-model上加：-->
</template>

<script setup lang="ts">
import { useMyStore } from "@/stores/store";
import { polarTopoDistAlgos, polarEmbDiffAlgos } from "@/stores/enums";
import type {
    AggregatedView,
    LinkableView,
    PolarView,
    PolarCoord,
} from "@/types/types";
import { default as setting } from "@/components/icon/FluentSettings48Regular.vue";
import { ref, watch, shallowRef, computed } from "vue";
import PolarRenderer from "./polarRenderer.vue";
import { rescalePolarCoords } from "@/utils/graphUtils";
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
const db = myStore.getCompDashboardById(props.dbId)!;
const view = myStore.getViewByName(props.dbId, props.viewName) as PolarView &
    LinkableView &
    AggregatedView;
view.isUsingSecondDbPredLabel = false;

const ds1 = myStore.getDatasetByName(db.refDatasetsNames[0])!;
const ds2 = myStore.getDatasetByName(db.refDatasetsNames[1])!;

const srcEntryIds = myStore.getViewSourceNodesSelectionEntry(props.viewName);

const singleSrcEntryId = srcEntryIds.find((d) => d.match(/single/i))!; //REVIEW
const multiSrcEntryId = srcEntryIds.find((d) => !d.match(/single/i))!;

const trueLabels = ds1?.trueLabels || ds2?.trueLabels || [];
const predLabels = computed(() =>
    view.isUsingSecondDbPredLabel ? ds2.predLabels || [] : ds1.predLabels || []
);

const opacitySliderValue = ref(20);
const handleOpacitySliderChange = (v: number) => {
    view.linkOpacity = v / 100;
};
watch(
    () => view.bodyComp,
    () => {
        view.linkOpacity = opacitySliderValue.value / 100;
    },
    {
        immediate: true, //NOTE: 这个watch是为了首次就能把opacity挂载到renderer
        flush: "post",
    }
);

const runAggregationJobs = async () => {
    db.polarWorker.workerTerminate();
    try {
        myStore.calcAggregatedViewData(
            view,
            (v) => v.cartesianCoords,
            () => [
                // [
                //     view.bodyMargins.left * view.bodyWidth - view.R,
                //     view.bodyMargins.top * view.bodyHeight - view.R,
                // ],
                // [
                //     view.R - view.bodyMargins.right * view.bodyWidth,
                //     0 - view.bodyMargins.bottom * view.bodyHeight,
                // ],
                [
                    view.bodyMargins.left * view.bodyWidth - view.bodyWidth / 2,
                    view.bodyMargins.top * view.bodyHeight -
                        view.bodyHeight / 2,
                ],
                [
                    view.bodyWidth - view.bodyMargins.right * view.bodyWidth,
                    view.bodyHeight - view.bodyMargins.bottom * view.bodyHeight,
                ],
            ],
            ds1.taskType === "node-classification"
                ? db.labelType === "true"
                    ? trueLabels
                    : predLabels.value
                : trueLabels,
            true,
            view.localLinks
        );
        return new Promise<void>((resolve, reject) => {
            view.bodyComp = shallowRef(PolarRenderer);
            resolve();
        });
    } catch (e) {
        return new Promise<void>((resolve, reject) => {
            view.bodyProps = { error: e };
            view.bodyComp = shallowRef(Error);
            reject();
        });
    }
};
watch(
    () => view.isShowAggregation, //NOTE 应当是在switch isAggregate 之后，立刻resize一次。
    (newV) => {
        // view.hideRectWhenClearSelFunc();
        if (newV) runAggregationJobs();
        // else { }//在render中实现了
    },
    {
        flush: "post", // NOTE post保证了计算aggregate时候，用的是新数据
    }
);
watch(
    () => view.meshRadius,
    () => {
        if (!view.isShowAggregation) {
            view.isShowAggregation = true;
        } else {
            runAggregationJobs();
        }
    },
    {
        flush: "post", // NOTE post保证了计算aggregate时候，用的是新半径
        // onTrigger(event) {
        //     console.warn(
        //         "in polarHeader, in watch view.meshRadius, triggered!",
        //         event
        //     );
        // },
    }
);
</script>

<style scoped>
.header {
    display: flex;
    flex-direction: row;
    flex-wrap: nowrap;
}
</style>
