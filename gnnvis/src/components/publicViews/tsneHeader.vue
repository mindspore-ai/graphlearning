<template>
    <span>
        <el-button @click="rootClick"> origin root tsne</el-button>
        <el-button @click="rerunClick">rerun local tsne</el-button>
        <el-button
            :loading="outlierLoading"
            type="primary"
            :disabled="
                viewDef.isShowAggregation ||
                tsneWorkerStatus === 'ERROR' ||
                tsneWorkerStatus === 'RUNNING' ||
                tsneWorkerStatus === 'TIMEOUT_EXPIRED'
            "
            @click="runOutlierDetection"
        >
            detect outlier</el-button
        >
    </span>
    <span>
        <el-switch
            v-model="viewDef.isShowAggregation"
            inactive-text="origin points"
            active-text="aggregated points"
        ></el-switch>
        <!-- :before-change="() => handleAggregateSwitch(false)"
            :loading="aggregateSwitchLoading" -->
        <!-- NOTE we handle those effects after boolean change -->
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
                :format-tooltip="(v:number)=>v/100"
                v-model="opacitySliderValue"
                @input="handleOpacitySliderChange"
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

            <span class="setting-item-name"
                >tsne hyperparameter:{{ " " }}
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
    LinkableView,
    NodeCoord,
    SingleDashboard,
    TsneCoord,
} from "@/types/types";
import { useWebWorkerFn } from "@/utils/myWebWorker";
import { calcTsne, rescaleCoords } from "@/utils/graphUtils";
import Loading from "../state/Loading.vue";
import Error from "../state/Error.vue";
import { default as setting } from "@/components/icon/FluentSettings48Regular.vue";
import TsneRenderer from "./tsneRenderer.vue";
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

viewDef.nodeRadius = 2;
viewDef.isShowHopSymbols = db.isRoot ? false : true;

const tgtEntryIds = myStore.getViewTargetNodesSelectionEntry(props.viewName);

// outlier detection
const outlierLoading = ref(false);
const runOutlierDetection = async (e: Event) => {
    outlierLoading.value = true;

    const data = viewDef.rescaledCoords;
    const meanX = data.reduce((sum, point) => sum + point.x, 0) / data.length;
    const stdDevX = Math.sqrt(
        data.reduce((sum, point) => sum + Math.pow(point.x - meanX, 2), 0) /
            data.length
    );

    const meanY = data.reduce((sum, point) => sum + point.y, 0) / data.length;
    const stdDevY = Math.sqrt(
        data.reduce((sum, point) => sum + Math.pow(point.y - meanY, 2), 0) /
            data.length
    );

    // 设置 Z-Score 阈值（通常是 2 或 3）
    const zScoreThreshold = 2;

    // 检测异常值
    const outliers = data.filter((point) => {
        const zScoreX = Math.abs((point.x - meanX) / stdDevX);
        const zScoreY = Math.abs((point.y - meanY) / stdDevY);
        return zScoreX > zScoreThreshold || zScoreY > zScoreThreshold;
    });

    console.log("in db", db.name, "in tsne view, Detected outliers:", outliers);
    tgtEntryIds.forEach((entryId) => {
        db.nodesSelections[entryId] = outliers.reduce(
            (acc, cur) => ({ ...acc, [cur.id]: db.srcNodesDict[cur.id] }),
            {}
        );
    });
    outlierLoading.value = false;
    myStore.repairButtonFocus(e);
};

//opacity related
const opacitySliderValue = ref(20);
const handleOpacitySliderChange = (v: number) => {
    if (viewDef.viewName.match(/latent/i)) {
        // viewDef.bodyProps = {
        //     ...viewDef.bodyProps,
        //     lineOpacity: v / 100,
        // };
        viewDef.linkOpacity = v / 100;
    }
};
watch(
    () => viewDef.bodyComp,
    (newV) => {
        if (viewDef.viewName.match(/latent/i)) {
            // console.warn(
            //     "in tsneHeader, () => viewDef.bodyComp, changed!",
            //     newV
            // );
            //以下两者皆可
            // viewDef.bodyProps = {
            //     ...viewDef.bodyProps,
            //     lineOpacity: opacitySliderValue.value / 100,
            // };
            // viewDef.bodyProps.lineOpacity = opacitySliderValue.value / 100;
            viewDef.linkOpacity = opacitySliderValue.value / 100;
        }
    },
    {
        immediate: true, //NOTE: 这个watch是为了首次就能把opacity挂载到renderer
        flush: "post",
    }
);

// rerun tsne related
const oldBodyProps = computed(() => ({
    which: props.which,
    // lineOpacity: opacitySliderValue.value / 10,
}));

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
    workerFn: tsneWorkerFn,
    workerStatus: tsneWorkerStatus,
    workerTerminate: tsneWorkerTerminate,
} = useWebWorkerFn(calcTsne, {
    timeout: 60_000,
    dependencies: ["http://localhost:5173/workers/tsne.js"],
});

const rootClick = (e: Event) => {
    //NOTE 用props.which或hasOwnProperty来判断db类型皆可
    const reject = Object.hasOwn(db, "tsneCalcPromise")
        ? (db as SingleDashboard).tsneCalcPromiseReject
        : Object.hasOwn(db, "tsne1CalcPromise")
        ? (db as CompDashboard).tsne1CalcPromiseReject
        : Object.hasOwn(db, "tsne2CalcPromise")
        ? (db as CompDashboard).tsne2CalcPromiseReject
        : () => {};
    reject("restore to root tsne ret!"); //拒绝rerun中的外层Promise
    tsneWorkerTerminate(); //拒绝内层，即worker

    if (props.which == 1) {
        const rootDb = myStore.compDashboardList.find((d) => d.isRoot);
        (db as CompDashboard).tsneRet1 =
            rootDb?.tsneRet1.filter((d) => db.srcNodesDict[d.id]) || [];
    } else if (props.which == 2) {
        const rootDb = myStore.compDashboardList.find((d) => d.isRoot);
        (db as CompDashboard).tsneRet2 =
            rootDb?.tsneRet2.filter((d) => db.srcNodesDict[d.id]) || [];
    } else {
        const rootDb = myStore.singleDashboardList.find((d) => d.isRoot);
        (db as SingleDashboard).tsneRet =
            rootDb?.tsneRet.filter((d) => db.srcNodesDict[d.id]) || [];
    }
    viewDef
        .setAttr("bodyComp", shallowRef(TsneRenderer))
        .setAttr("bodyProps", oldBodyProps.value); //NOTE 额外的props是通过v-bind传递给bodyProps的
    viewDef.isShowAggregation = false;
    myStore.repairButtonFocus(e);
};

const rerunClick = (e: Event | undefined) => {
    const reject = Object.hasOwn(db, "tsneCalcPromise")
        ? (db as SingleDashboard).tsneCalcPromiseReject
        : Object.hasOwn(db, "tsne1CalcPromise")
        ? (db as CompDashboard).tsne1CalcPromiseReject
        : Object.hasOwn(db, "tsne2CalcPromise")
        ? (db as CompDashboard).tsne2CalcPromiseReject
        : () => {};
    reject("in " + props.viewName + " rerun tsne!");
    tsneWorkerTerminate();

    /////////////////////////////////////////
    ///// calc dimension reduction

    const reducedRawEmb =
        toRaw(ds?.embNode)
            ?.map((d, i) => ({ id: i + "", emb: d }))
            .filter((d) => db.srcNodesDict[d.id])
            .sort((a, b) => a.id.localeCompare(b.id)) || [];
    // const oldProps = viewDef.bodyProps; //保存之前的 //NOTE 额外的props是通过v-bind传递给bodyProps的,且每次setAttr都是replace，要注意！

    console.warn(
        "in rerun tsne, ready to set viewDef.bodyComp, viewDef now is",
        viewDef
    );
    viewDef.bodyComp = shallowRef(Loading);
    viewDef.bodyProps = { text: "rerunning tsne" };

    if (props.which == 1) {
        (db as CompDashboard).tsne1CalcPromise = new Promise<void>(
            (resolve, reject) => {
                (db as CompDashboard).tsne1CalcPromiseReject = reject;
                tsneWorkerFn(
                    reducedRawEmb.map((d) => d.emb),
                    iterCount.value,
                    perplexity.value,
                    epsilon.value
                )
                    .then((data) => {
                        (db as CompDashboard).tsneRet1 = data.map((d, i) => ({
                            id: reducedRawEmb[i].id,
                            x: d[0],
                            y: d[1],
                        }));
                        tsneWorkerTerminate();
                        viewDef
                            .setAttr("bodyComp", shallowRef(TsneRenderer))
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
        (db as CompDashboard).tsne1CalcPromise = new Promise<void>(
            (resolve, reject) => {
                (db as CompDashboard).tsne2CalcPromiseReject = reject;
                tsneWorkerFn(
                    reducedRawEmb.map((d) => d.emb),
                    iterCount.value,
                    perplexity.value,
                    epsilon.value
                )
                    .then((data) => {
                        (db as CompDashboard).tsneRet2 = data.map((d, i) => ({
                            id: reducedRawEmb[i].id,
                            x: d[0],
                            y: d[1],
                        }));
                        tsneWorkerTerminate();
                        viewDef
                            // .setAttr("isShowAggregation", false)
                            .setAttr("bodyComp", shallowRef(TsneRenderer))
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
        (db as SingleDashboard).tsneCalcPromise = new Promise<void>(
            (resolve, reject) => {
                (db as SingleDashboard).tsneCalcPromiseReject = reject;
                tsneWorkerFn(
                    reducedRawEmb.map((d) => d.emb),
                    iterCount.value,
                    perplexity.value,
                    epsilon.value
                )
                    .then((data) => {
                        console.warn("in rerun tsne, worker.then, db is", db);
                        console.warn(
                            "in rerun tsne, worker.then, data is",
                            data
                        );
                        (db as SingleDashboard).tsneRet = data.map((d, i) => ({
                            id: reducedRawEmb[i].id,
                            x: d[0],
                            y: d[1],
                        }));
                        tsneWorkerTerminate();
                        viewDef
                            // .setAttr("isShowAggregation", false)
                            .setAttr("bodyComp", shallowRef(TsneRenderer))
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

const trueLabels = ds.trueLabels || [];
const predLabels = ds.predLabels || [];
const runAggregationJobs = async () => {
    const reject = Object.hasOwn(db, "tsneCalcPromise")
        ? (db as SingleDashboard).tsneCalcPromiseReject
        : Object.hasOwn(db, "tsne1CalcPromise")
        ? (db as CompDashboard).tsne1CalcPromiseReject
        : Object.hasOwn(db, "tsne2CalcPromise")
        ? (db as CompDashboard).tsne2CalcPromiseReject
        : () => {};
    reject("in " + props.viewName + " aggregate calc encountered");
    tsneWorkerTerminate();
    try {
        //     aggregateSwitchLoading.value = false;
        // await new Promise<void>((resolve, reject) => {
        //     viewDef.bodyProps = { text: "calc aggregated coords" };
        //     viewDef.bodyComp = shallowRef(Loading);
        //     setTimeout(() => {
        //         resolve();
        //     }, 1500);
        // });
        // await nextTick();

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
            ds.taskType === "node-classification"
                ? db.labelType === "true"
                    ? trueLabels
                    : predLabels
                : trueLabels,
            true,
            db.srcLinksArr
        );
        return new Promise<void>((resolve, reject) => {
            // setTimeout(() => {
            viewDef.bodyProps = oldBodyProps.value;
            viewDef.bodyComp = shallowRef(TsneRenderer);
            resolve();
            // }, 2000);
        });
        // return new Promise<boolean>((resolve, reject) => {
        //     // setTimeout(() => {
        //     resolve(true);
        //     // }, 2000);
        // });
    } catch (e) {
        return new Promise<void>((resolve, reject) => {
            viewDef.bodyProps = { error: e };
            viewDef.bodyComp = shallowRef(Error);
            reject();
        });
    }
    // finally {
    //     aggregateSwitchLoading.value = false;
    // }
};
watch(
    [() => viewDef.isShowAggregation, () => db.labelType], //NOTE 应当是在switch isAggregate 之后，立刻resize一次。
    ([newV, newLabelType]) => {
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
                (d: TsneCoord) => d.x,
                (d: TsneCoord) => d.y,
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
        //     .setAttr("bodyComp", shallowRef(TsneRenderer))
        //     .setAttr("bodyProps", oldBodyProps.value);
    },
    {
        flush: "post", // NOTE post保证了计算aggregate时候，用的是新半径
        // onTrigger(event) {
        //     console.warn(
        //         "in tsneHeader, in watch viewDef.meshRadius, triggered!",
        //         event
        //     );
        // },
    }
);
</script>

<style scoped></style>
