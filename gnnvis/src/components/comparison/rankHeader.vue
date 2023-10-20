<template>
    <el-tooltip
        effect="dark"
        placement="top"
        content="select some nodes and calc rank first!"
        :disabled="db.rankWorker.workerStatus === 'SUCCESS'"
    >
        <el-switch
            :disabled="db.rankWorker.workerStatus !== 'SUCCESS'"
            v-model="view.isShowAggregation"
            inactive-text="origin points"
            active-text="aggregated points"
        ></el-switch>
    </el-tooltip>

    <el-tooltip effect="dark" placement="top">
        <template #content>
            create a temp polar view
            <br />
            only for this view
        </template>
        <el-button
            :disabled="isEmptyDict(db.nodesSelections[tgtEntryIds[0]])"
            @click="handleCreatePolar"
        >
            <createPolarSvg />
        </el-button>
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
            <h3>Comparative Rank View Settings</h3>
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
                :disabled="db.rankWorker.workerStatus !== 'SUCCESS'"
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

            <span class="setting-item-name"> emb counting algo{{ " " }} </span>
            <el-tooltip effect="dark" placement="top">
                <template #content>
                    the algorithm of embs accounting
                    <br />
                    if single node selected, simply calc all distances from 'all
                    nodes' to selected node
                    <br />
                    center: treat nodes as a cluster, and calc all distances
                    from 'all nodes' to cluster center
                    <br />
                    avg: for one node in 'all nodes', calc avg dist to selected
                    nodes, and repeat for 'all nodes'
                </template>
                <el-select
                    :style="{ width: 100 + 'px' }"
                    v-model="view.rankEmbDiffAlgo"
                    clearable
                    placeholder="select algorithm"
                >
                    <el-option
                        v-for="a in rankEmbDiffAlgos"
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
                :disabled="view.rankEmbDiffAlgo === 'single'"
            >
                <template #content> only available in "single" algo </template>
                <el-select
                    :disabled="view.rankEmbDiffAlgo !== 'single'"
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

            <br />
        </div>
    </el-popover>
</template>

<script setup lang="ts">
import { useMyStore } from "@/stores/store";
import { rankEmbDiffAlgos } from "@/stores/enums";
import { default as createPolarSvg } from "../icon/LucideTarget.vue";
import {
    shallowRef,
    ref,
    defineAsyncComponent,
    computed,
    watch,
    h,
    toRaw,
    nextTick,
} from "vue";
import { useWebWorkerFn } from "@/utils/myWebWorker";
import { calcPolar, isEmptyDict, rescaleCoords } from "@/utils/graphUtils";
import PolarHeader from "./polarHeader.vue";
import LoadingComp from "../state/Loading.vue";
import ErrorComp from "../state/Error.vue";
import { default as setting } from "@/components/icon/FluentSettings48Regular.vue";
import { ElButton } from "element-plus";
import type {
    CompDashboard,
    NodeCoord,
    PolarView,
    RankView,
} from "@/types/types";
import RankRenderer from "./rankRenderer.vue";
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
const db = myStore.getCompDashboardById(props.dbId) as CompDashboard;
const view = myStore.getViewByName(props.dbId, props.viewName) as RankView;
view.isUsingSecondDbPredLabel = false;

const ds1 = myStore.getDatasetByName(db.refDatasetsNames[0])!;
const ds2 = myStore.getDatasetByName(db.refDatasetsNames[1])!;

const trueLabels = ds1.trueLabels || ds2.trueLabels || [];
const predLabels = computed(() =>
    view.isUsingSecondDbPredLabel ? ds2.predLabels || [] : ds1.predLabels || []
);
const tgtEntryIds = myStore.getViewTargetNodesSelectionEntry(props.viewName);

const handleCreatePolar = async (e: Event) => {
    const newViewName = "Polar View Affiliated to " + props.viewName;

    if (myStore.getViewByName(props.dbId, newViewName)) {
        myStore.repairButtonFocus(e);
        return; //单例模式
    }

    //src和tgt的nodesSelectionEntry的定义加入
    myStore.defaultNodesSelectionEntryMapper[newViewName] = {
        source: ["rankOut", "rankOutSingle"],
        target: ["comparativeOut2"],
    };
    myStore.defaultNodesSelectionEntryDescriptions["comparativeOut2"] =
        "selections from (" + newViewName + ")";
    db.nodesSelections["comparativeOut2"] = {}; //REVIEW setDbSelectionEntry ?

    //加入viewList
    const newPolarView = myStore.initialNewView(newViewName) as PolarView;
    const PolarComp = shallowRef<any>(null); //如果不想中间再跨一个到达svg，那就用这个//REVIEW :idle comp?
    const polarProps = ref({});
    myStore
        .insertViewAfterName(props.dbId, props.viewName, newPolarView)
        // myStore
        //     .getViewByName(db, newViewName)
        ?.setAttr("headerComp", shallowRef(PolarHeader)) //REVIEW 凭什么这里能写PolarHeader？而不是用一个抽象的方法
        .setAttr("bodyComp", PolarComp)
        .setAttr("bodyProps", polarProps)
        .setAttr("initialWidth", 800);

    //start calculation
    const hops = ds1.hops || ds2.hops || myStore.defaultHops;
    const nodeMapLink = ds1.nodeMapLink || ds2.nodeMapLink || [];
    const neighborMasksByHop =
        ds1.neighborMasksByHop || ds2.neighborMasksByHop || [];

    const polarEmbDiffAlgo = computed(
        () => myStore.getViewByName(props.dbId, newViewName)?.polarEmbDiffAlgo!
    );
    const polarTopoDistAlgo = computed(
        () => myStore.getViewByName(props.dbId, newViewName)?.polarTopoDistAlgo!
    );

    const singleSel = computed(
        () => db.nodesSelections["rankOutSingle"] || {} //NOTE 这里已经有具体字符串了，就是上面那个rankOut啥的
    );
    const multiSel = computed(() => db.nodesSelections["rankOut"] || {});

    const calcPolarError = ref<ErrorEvent | Error>();
    const calcPolarRet = ref<Array<any>>();
    const {
        workerFn: calcPolarWorkerFn,
        workerStatus: calcPolarWorkerStatus, //创建时即pending
        workerTerminate: calcPolarWorkerTerminate,
    } = useWebWorkerFn(calcPolar, {
        //NOTE： 此创建不可重用，因为返回的status是ref，若重用则共享状态，出现混乱。
        timeout: 20_000,
        dependencies: [
            // "https://d3js.org/d3.v7.min.js",
            "http://localhost:5173/workers/d3js.org_d3.v7.js",
            "http://localhost:5173/workers/bitset.js",
            "http://localhost:5173/workers/distance.js", // REVIEW temporary
        ],
    });

    const doCalcPolar = async (
        [newEmbAlgo, newTopoAlgo, newId, newDict]: [
            "single" | "average" | "center" | undefined,
            "shortest path" | "hamming" | "jaccard" | undefined,
            Record<string, boolean>, //实际运行中newId往往是string即只有键
            Record<string, boolean>
        ],
        [oldEmbAlgo, oldTopoAlgo, oldId, oldDict]: [
            "single" | "average" | "center" | undefined,
            "shortest path" | "hamming" | "jaccard" | undefined,
            Record<string, boolean>,
            Record<string, boolean>
        ] = [undefined, undefined, {}, {}]
    ) => {
        console.log(
            "in",
            newViewName,
            "in doCalcPolar, new and old: ",
            [newEmbAlgo, newTopoAlgo, newId, newDict],
            [oldEmbAlgo, oldTopoAlgo, oldId, oldDict]
        );
        if (calcPolarWorkerStatus.value === "RUNNING") {
            calcPolarWorkerTerminate();
        }
        calcPolarError.value = undefined;
        try {
            calcPolarRet.value = await calcPolarWorkerFn(
                newTopoAlgo!,
                newEmbAlgo!,
                newEmbAlgo === "single"
                    ? toRaw(singleSel.value)
                    : toRaw(multiSel.value),
                toRaw(ds1.embNode),
                toRaw(ds2.embNode),
                toRaw(nodeMapLink),
                hops,
                toRaw(neighborMasksByHop)
            );
            console.log(
                "in",
                newViewName,
                "in doCalcPolar got ret",
                calcPolarRet.value
            );
        } catch (e) {
            //reject
            calcPolarError.value = e as Error | ErrorEvent;
            console.log(
                "in",
                newViewName,
                "now in doCalcPolar we catch an error",
                e
            );
        }
    };
    const affiliatedPolarCalcWatcher = watch(
        [polarEmbDiffAlgo, polarTopoDistAlgo, singleSel, multiSel],
        doCalcPolar,
        {
            deep: true,
            immediate: true, //NOTE 这里必须是true，因为我们要立刻看结果
            // onTrigger(event) {
            //     console.warn(
            //         "in ",
            //         newViewName,
            //         " calc polar watch triggered, event:",
            //         event
            //     );
            // },//NOTE only for debug, only in dev mode
        }
    );

    const affiliatedPolarCompWatcher = watch(
        calcPolarWorkerStatus,
        (newV) => {
            if (newV === "PENDING" || newV === "RUNNING") {
                polarProps.value = { text: calcPolarWorkerStatus.value };
                PolarComp.value = LoadingComp;
            } else if (newV === "SUCCESS") {
                polarProps.value = { data: calcPolarRet.value, hops: hops };
                PolarComp.value = defineAsyncComponent({
                    loader: async () => {
                        return await import("./polarRenderer.vue");
                    },
                    delay: 200,
                    timeout: 1000,
                    loadingComponent: LoadingComp,
                    errorComponent: ErrorComp,
                });
            } else {
                polarProps.value = { error: calcPolarError.value?.message };
                PolarComp.value = h("div", [
                    //懒得再写一个组件了，直接上渲染函数
                    h(ErrorComp, polarProps.value),
                    h(
                        ElButton,
                        {
                            onClick: (e: Event) => {
                                doCalcPolar([
                                    polarEmbDiffAlgo.value,
                                    polarTopoDistAlgo.value,
                                    singleSel.value,
                                    multiSel.value,
                                ]);
                                myStore.repairButtonFocus(e);
                            },
                        },
                        () => "retry" //Non-function value encountered for default slot. Prefer function slots for better performance.
                    ),
                ]);
            }
        },
        { immediate: true }
    );

    myStore.getViewByName(db, newViewName)!.onBeforeUnmountCallbacks = [
        affiliatedPolarCalcWatcher, //清除侦听器的函数
        affiliatedPolarCompWatcher,
        () => {
            console.log("in view", newViewName, "Unmounted callbacks!");
        },
    ];

    myStore.repairButtonFocus(e);
};

const runAggregationJobs = async () => {
    db.rankWorker.workerTerminate();
    try {
        myStore.calcAggregatedViewData(
            view,
            (v) => v.rescaledCoords,
            () => [
                [
                    view.bodyWidth * view.bodyMargins.left,
                    view.bodyHeight * (1 - view.bodyMargins.bottom),
                ],
                [
                    view.bodyWidth * (1 - view.bodyMargins.right),
                    view.bodyHeight * view.bodyMargins.top,
                ],
            ],
            ds1.taskType === "node-classification"
                ? db.labelType === "true"
                    ? trueLabels
                    : predLabels.value
                : trueLabels,
            false,
            []
        );
        return new Promise<void>((resolve, reject) => {
            // setTimeout(() => {
            view.bodyComp = shallowRef(RankRenderer);
            resolve();
            // }, 2000);
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
        else {
            view.rescaledCoords = rescaleCoords<NodeCoord, never>(
                view.sourceCoords as NodeCoord[],
                [
                    view.bodyWidth * view.bodyMargins.left,
                    view.bodyWidth * (1 - view.bodyMargins.right),
                ],
                [
                    view.bodyHeight * (1 - view.bodyMargins.bottom),
                    view.bodyHeight * view.bodyMargins.top,
                ],
                (d: NodeCoord) => d.x,
                (d: NodeCoord) => d.y,
                db.name + "===" + props.viewName
            );
        }
    },
    {
        flush: "post",
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
        //         "in rankHeader, in watch view.meshRadius, triggered!",
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
    align-items: center;
    flex-wrap: nowrap;
}
</style>
