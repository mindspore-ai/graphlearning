<template>
    <span>
        <el-button @click="rootClick"> origin root graphTsne</el-button>
        <el-button @click="rerunClick">rerun local graphTsne</el-button>
    </span>
    <span>
        <el-switch
            v-model="viewDef.isShowAggregation"
            inactive-text="origin points"
            active-text="aggregated points"
        ></el-switch>
    </span>

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
            <h3>Tsne View Settings</h3>
            <el-divider />

            <span class="setting-item-name">point radius {{ " " }} </span>
            <el-slider
                v-model="viewDef.nodeRadius"
                :min="2"
                :step="0.1"
                :max="
                    Math.max(
                        Math.min(viewDef.bodyWidth, viewDef.bodyHeight) / 80,
                        10
                    )
                "
            />
            <br />

            <span class="setting-item-name"
                >aggregate hexbin radius:{{ " " }}
            </span>
            <el-slider
                :min="10"
                :max="100"
                v-model="viewDef.meshRadius"
            ></el-slider>
            <br />

            <span class="setting-item-name"> if show mesh:{{ " " }} </span>
            <el-tooltip
                effect="dark"
                placement="top"
                :disabled="Boolean(viewDef.hexbin)"
                content="run aggregation first to show mesh!"
            >
                <el-switch
                    :disabled="!Boolean(viewDef.hexbin)"
                    v-model="viewDef.isShowMesh"
                    inactive-text="hide mesh"
                    active-text="show mesh"
                ></el-switch>
            </el-tooltip>
            <br />

            <span class="setting-item-name"
                >graphTsne hyperparameter:{{ " " }}
            </span>
            <br />
            <span class="setting-item-name">perplexity:{{ " " }} </span>
            <el-slider
                :width="100"
                v-model="perplexity"
                show-input
                :min="5"
                :max="50"
            />
            <br />
            <span class="setting-item-name">iter:{{ " " }} </span>
            <el-slider v-model="iterCount" show-input :min="1" :max="5000" />
            <br />
            <span class="setting-item-name">epsilon(lr):{{ " " }} </span>
            <el-slider v-model="epsilon" show-input :min="10" :max="1000" />
        </div>
    </el-popover>
</template>

<script setup lang="ts">
import { toRaw, shallowRef, ref, watch, computed } from "vue";
import { useMyStore } from "@/stores/store";
import type {
    AggregatedView,
    CompDashboard,
    GraphTsneCoord,
    LinkableView,
    NodeCoord,
    SingleDashboard,
    TsneCoord,
} from "@/types/types";
import { useWebWorkerFn } from "@/utils/myWebWorker";
import {
    calcTsne,
    nodeMapGraph2GraphMapNodes,
    rescaleCoords,
} from "@/utils/graphUtils";
import Loading from "../state/Loading.vue";
import Error from "../state/Error.vue";
import { default as setting } from "@/components/icon/FluentSettings48Regular.vue";
import GraphTsneRenderer from "./graphTsneRenderer.vue";
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
    which: {
        type: Number,
        required: true,
    },
});
const myStore = useMyStore();
const viewDef = myStore.getViewByName(
    props.dbId,
    props.viewName
) as AggregatedView & LinkableView;
const db = myStore.getTypeReducedDashboardById(props.dbId)!;
const ds =
    props.which == 1
        ? myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[0])!
        : props.which == 2
        ? myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[1])!
        : myStore.getDatasetByName((db as SingleDashboard).refDatasetName)!;

const oldBodyProps = computed(() => ({
    which: props.which,
}));

viewDef.nodeRadius = 2;

const srcGraphsDict = computed(() =>
    nodeMapGraph2GraphMapNodes(db.srcNodesDict)
);

const iterCount = ref(5000);
const perplexity = ref(30);
const epsilon = ref(10);
let timer = -1;
watch(
    [iterCount, perplexity, epsilon],
    () => {
        clearTimeout(timer);
        timer = setTimeout(() => {
            rerunClick(undefined);
        }, 300);
    },
    {
        flush: "post",
    }
);
const {
    workerFn: graphTsneWorkerFn,
    // workerStatus: graphTsneWorkerStatus,
    workerTerminate: graphTsneWorkerTerminate,
} = useWebWorkerFn(calcTsne, {
    timeout: 60_000,
    dependencies: ["http://localhost:5173/workers/tsne.js"],
});

const rootClick = (e: Event) => {
    //NOTE 用props.which或hasOwnProperty来判断db类型皆可
    const reject = Object.hasOwn(db, "graphTsneCalcPromise")
        ? (db as SingleDashboard).graphTsneCalcPromiseReject
        : Object.hasOwn(db, "graphTsne1CalcPromise")
        ? (db as CompDashboard).graphTsne1CalcPromiseReject
        : Object.hasOwn(db, "graphTsne2CalcPromise")
        ? (db as CompDashboard).graphTsne2CalcPromiseReject
        : () => {};
    reject("restore to root graphTsne ret!"); //拒绝rerun中的外层Promise
    graphTsneWorkerTerminate(); //拒绝内层，即worker

    if (props.which == 1) {
        const rootDb = myStore.compDashboardList.find((d) => d.isRoot);
        (db as CompDashboard).graphTsneRet1 =
            rootDb?.graphTsneRet1.filter((d) => srcGraphsDict.value[d.id]) ||
            [];
    } else if (props.which == 2) {
        const rootDb = myStore.compDashboardList.find((d) => d.isRoot);
        (db as CompDashboard).graphTsneRet2 =
            rootDb?.graphTsneRet2.filter((d) => srcGraphsDict.value[d.id]) ||
            [];
    } else {
        const rootDb = myStore.singleDashboardList.find((d) => d.isRoot);
        (db as SingleDashboard).graphTsneRet =
            rootDb?.graphTsneRet.filter((d) => srcGraphsDict.value[d.id]) || [];
    }
    viewDef
        .setAttr("bodyComp", shallowRef(GraphTsneRenderer))
        .setAttr("bodyProps", oldBodyProps.value); //NOTE 额外的props是通过v-bind传递给bodyProps的
    viewDef.isShowAggregation = false;
    myStore.repairButtonFocus(e);
};

const rerunClick = (e: Event | undefined) => {
    const reject = Object.hasOwn(db, "graphTsneCalcPromise")
        ? (db as SingleDashboard).graphTsneCalcPromiseReject
        : Object.hasOwn(db, "graphTsne1CalcPromise")
        ? (db as CompDashboard).graphTsne1CalcPromiseReject
        : Object.hasOwn(db, "graphTsne2CalcPromise")
        ? (db as CompDashboard).graphTsne2CalcPromiseReject
        : () => {};
    reject("in " + props.viewName + " rerun graphTsne!");
    graphTsneWorkerTerminate();

    /////////////////////////////////////////
    ///// calc dimension reduction

    const reducedRawEmb =
        toRaw(ds?.embGraph)
            ?.map((d, i) => ({ id: i + "", emb: d }))
            .filter((d) => srcGraphsDict.value[d.id])
            .sort((a, b) => a.id.localeCompare(b.id)) || [];

    viewDef.bodyComp = shallowRef(Loading);
    viewDef.bodyProps = { text: "rerunning graphTsne" };

    if (props.which == 1) {
        (db as CompDashboard).graphTsne1CalcPromise = new Promise<void>(
            (resolve, reject) => {
                (db as CompDashboard).graphTsne1CalcPromiseReject = reject;
                graphTsneWorkerFn(
                    reducedRawEmb.map((d) => d.emb),
                    iterCount.value,
                    perplexity.value,
                    epsilon.value
                )
                    .then((data) => {
                        (db as CompDashboard).graphTsneRet1 = data.map(
                            (d, i) => ({
                                id: reducedRawEmb[i].id,
                                x: d[0],
                                y: d[1],
                            })
                        );
                        graphTsneWorkerTerminate();
                        viewDef
                            .setAttr("bodyComp", shallowRef(GraphTsneRenderer))
                            .setAttr("bodyProps", oldBodyProps.value); //NOTE 额外的props是通过v-bind传递给bodyProps的,且每次setAttr都是replace，要注意！
                        viewDef.isShowAggregation = false;
                        resolve();
                    })
                    .catch((e) => {
                        reject(e);
                        viewDef
                            .setAttr("bodyProps", { error: e })
                            .setAttr("bodyComp", Error);
                    });
            }
        );
    } else if (props.which == 2) {
        (db as CompDashboard).graphTsne1CalcPromise = new Promise<void>(
            (resolve, reject) => {
                (db as CompDashboard).graphTsne2CalcPromiseReject = reject;
                graphTsneWorkerFn(
                    reducedRawEmb.map((d) => d.emb),
                    iterCount.value,
                    perplexity.value,
                    epsilon.value
                )
                    .then((data) => {
                        (db as CompDashboard).graphTsneRet2 = data.map(
                            (d, i) => ({
                                id: reducedRawEmb[i].id,
                                x: d[0],
                                y: d[1],
                            })
                        );
                        graphTsneWorkerTerminate();
                        viewDef
                            // .setAttr("isShowAggregation", false)
                            .setAttr("bodyComp", shallowRef(GraphTsneRenderer))
                            .setAttr("bodyProps", oldBodyProps.value); //NOTE 额外的props是通过v-bind传递给bodyProps的
                        viewDef.isShowAggregation = false;
                        resolve();
                    })
                    .catch((e) => {
                        reject(e);
                        viewDef
                            .setAttr("bodyProps", { error: e })
                            .setAttr("bodyComp", Error);
                    });
            }
        );
    } else {
        (db as SingleDashboard).graphTsneCalcPromise = new Promise<void>(
            (resolve, reject) => {
                (db as SingleDashboard).graphTsneCalcPromiseReject = reject;
                graphTsneWorkerFn(
                    reducedRawEmb.map((d) => d.emb),
                    iterCount.value,
                    perplexity.value,
                    epsilon.value
                )
                    .then((data) => {
                        console.warn(
                            "in rerun graphTsne, worker.then, db is",
                            db
                        );
                        console.warn(
                            "in rerun graphTsne, worker.then, data is",
                            data
                        );
                        (db as SingleDashboard).graphTsneRet = data.map(
                            (d, i) => ({
                                id: reducedRawEmb[i].id,
                                x: d[0],
                                y: d[1],
                            })
                        );
                        graphTsneWorkerTerminate();
                        viewDef
                            // .setAttr("isShowAggregation", false)
                            .setAttr("bodyComp", shallowRef(GraphTsneRenderer))
                            .setAttr("bodyProps", oldBodyProps.value); //NOTE 额外的props是通过v-bind传递给bodyProps的
                        viewDef.isShowAggregation = false;
                        resolve();
                    })
                    .catch((e) => {
                        reject(e);
                        viewDef
                            .setAttr("bodyProps", { error: e })
                            .setAttr("bodyComp", Error);
                    });
            }
        );
    }
    ///// calc dimension reduction
    /////////////////////////////////////////
    if (e) myStore.repairButtonFocus(e);
};

const trueLabels = ds.graphTrueLabels || [];
const predLabels = ds.graphPredLabels || [];
const runAggregationJobs = async () => {
    const reject = Object.hasOwn(db, "graphTsneCalcPromise")
        ? (db as SingleDashboard).graphTsneCalcPromiseReject
        : Object.hasOwn(db, "graphTsne1CalcPromise")
        ? (db as CompDashboard).graphTsne1CalcPromiseReject
        : Object.hasOwn(db, "graphTsne2CalcPromise")
        ? (db as CompDashboard).graphTsne2CalcPromiseReject
        : () => {};
    reject("in " + props.viewName + " aggregate calc encountered");
    graphTsneWorkerTerminate();
    try {
        myStore.calcAggregatedViewData(
            viewDef,
            (v) => v.rescaledCoords,
            () => [
                [
                    viewDef.bodyWidth * viewDef.bodyMargins.left,
                    viewDef.bodyHeight * viewDef.bodyMargins.top,
                ],
                [
                    viewDef.bodyWidth * (1 - viewDef.bodyMargins.right),
                    viewDef.bodyHeight * (1 - viewDef.bodyMargins.bottom),
                ],
            ], //extentFunction, using default
            db.labelType === "true" ? trueLabels : predLabels,
            false,
            []
        );
        return new Promise<void>((resolve, reject) => {
            // setTimeout(() => {
            viewDef.bodyProps = oldBodyProps.value;
            viewDef.bodyComp = shallowRef(GraphTsneRenderer);
            resolve();
            // }, 2000);
        });
    } catch (e) {
        return new Promise<void>((resolve, reject) => {
            viewDef.bodyProps = { error: e };
            viewDef.bodyComp = shallowRef(Error);
            reject();
        });
    }
};

watch(
    [() => viewDef.isShowAggregation, () => db.labelType], //NOTE 应当是在switch isAggregate 之后，立刻resize一次。
    ([newV, newLabel]) => {
        // viewDef.hideRectWhenClearSelFunc();
        if (newV) runAggregationJobs();
        else {
            viewDef.rescaledCoords = rescaleCoords(
                viewDef.sourceCoords,
                [
                    viewDef.bodyWidth * viewDef.bodyMargins.left,
                    viewDef.bodyWidth * (1 - viewDef.bodyMargins.right),
                ],
                [
                    viewDef.bodyHeight * viewDef.bodyMargins.top,
                    viewDef.bodyHeight * (1 - viewDef.bodyMargins.bottom),
                ],
                (d: GraphTsneCoord) => d.x,
                (d: GraphTsneCoord) => d.y,
                props.viewName
            );
        }
    },
    {
        flush: "post",
    }
);
watch(
    () => viewDef.meshRadius,
    () => {
        // viewDef.isShowAggregation = await handleAggregateSwitch();
        if (!viewDef.isShowAggregation) {
            viewDef.isShowAggregation = true;
        } else {
            runAggregationJobs();
        }
        // viewDef
        //     .setAttr("bodyComp", shallowRef(GraphTsneRenderer))
        //     .setAttr("bodyProps", oldBodyProps.value);
    },
    {
        flush: "post", // NOTE post保证了计算aggregate时候，用的是新半径
        // onTrigger(event) {
        //     console.warn(
        //         "in graphTsneHeader, in watch viewDef.meshRadius, triggered!",
        //         event
        //     );
        // },
    }
);
</script>

<style scoped></style>
