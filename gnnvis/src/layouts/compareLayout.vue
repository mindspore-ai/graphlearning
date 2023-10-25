<template>
    <div class="container">
        <div class="header-all">
            <h1 ref="h1Ref">GorGIE-2 Comparative</h1>
            <div class="header-flex">
                <div>
                    <RouterLink to="/home" :replace="true">
                        <el-button>Back to Home</el-button></RouterLink
                    >
                </div>
                <el-popover
                    placement="bottom-end"
                    :trigger="myStore.settingsMenuTriggerMode"
                    :width="400"
                >
                    <template #reference>
                        <setting
                            style="
                                height: 2em;
                                width: 2em;
                                color: var(--el-text-color-primary);
                            "
                        />
                    </template>
                    <div style="text-align: start; min-height: 250px">
                        <h3>CorGIE-2 Global Settings</h3>
                        <el-divider />

                        <span class="setting-item-name"
                            >Global label type:{{ " " }}
                        </span>
                        <el-tooltip
                            :effect="currentDsTaskType ? 'dark' : 'light'"
                            placement="top"
                            :z-index="9999"
                        >
                            <template #content>
                                <div
                                    v-if="
                                        currentDsTaskType ===
                                        'node-classification'
                                    "
                                >
                                    true or pred label for each node
                                </div>
                                <div
                                    v-else-if="
                                        currentDsTaskType === 'link-prediction'
                                    "
                                >
                                    disabled, since all node labels are true
                                    labels
                                </div>
                                <div
                                    v-else-if="
                                        currentDsTaskType ===
                                        'graph-classification'
                                    "
                                >
                                    true or pred label for each graph,<br />
                                    all node labels are true labels
                                </div>
                                <div v-else :style="{ color: 'red' }">
                                    waiting for fetching dataset...
                                </div>
                            </template>
                            <el-switch
                                :disabled="
                                    !currentDsTaskType ||
                                    currentDsTaskType === 'link-prediction'
                                "
                                v-model="myStore.globalLabelType"
                                active-text="pred label"
                                inactive-text="true label"
                                :active-value="'pred'"
                                :inactive-value="'true'"
                            />
                        </el-tooltip>
                        <br />

                        <span class="setting-item-name"
                            >Default(Global) select mode:{{ " " }}
                        </span>
                        <el-switch
                            v-model="myStore.globalClearSelMode"
                            :active-value="clearSelModes[0]"
                            :inactive-value="clearSelModes[1]"
                            active-text="manual clear (merge select)"
                            inactive-text="auto clear (re-select)"
                        >
                        </el-switch>
                        <br />

                        <span class="setting-item-name"
                            >Default(Global) Rescale coords when:{{ " " }}
                        </span>
                        <el-switch
                            v-model="myStore.globalWhenToResizeMode"
                            :active-value="whenToRescaleModes[1]"
                            :inactive-value="whenToRescaleModes[0]"
                            active-text="on resize end"
                            inactive-text="simultaneously"
                        >
                        </el-switch>
                        <br />
                        <br />

                        <span class="setting-item-name"
                            >New dashboard placement:{{ " " }}</span
                        >
                        <el-popconfirm
                            confirmButtonText="Yes"
                            cancelButtonText="No"
                            title="This will cause rerender!"
                            @confirm="confirmIsAppend"
                            @cancel="cancelIsAppend"
                            :width="200"
                        >
                            <template #reference>
                                <el-switch
                                    v-model="myStore.dashboardsLayoutMode"
                                    :inactive-text="dashboardsLayoutModes[1]"
                                    :active-text="dashboardsLayoutModes[0]"
                                    :inactive-value="dashboardsLayoutModes[1]"
                                    :active-value="dashboardsLayoutModes[0]"
                                    :before-change="beforeIsAppendChange"
                                >
                                </el-switch>
                            </template>
                        </el-popconfirm>
                        <br />

                        <span class="setting-item-name"
                            >Max recent length:{{ myStore.recentLen }}</span
                        >
                        <el-slider
                            v-model="myStore.recentLen"
                            :step="1"
                            show-stops
                            :min="myStore.minRecentLen"
                            :max="myStore.maxRecentLen"
                        >
                        </el-slider>
                        <br />

                        <span class="setting-item-name"
                            >Settings trigger mode:{{ " " }}</span
                        >
                        <el-switch
                            v-model="myStore.settingsMenuTriggerMode"
                            :inactive-text="settingsMenuTriggerModes[1]"
                            :active-text="settingsMenuTriggerModes[0]"
                            :inactive-value="settingsMenuTriggerModes[1]"
                            :active-value="settingsMenuTriggerModes[0]"
                        >
                        </el-switch>
                    </div>
                </el-popover>
            </div>
        </div>
        <el-scrollbar>
            <div class="main">
                <LoadingComp v-if="isLoading" :text="loadingText" />
                <ErrorComp v-else-if="loadingError" :error="loadingError" />
                <!-- component
                :is="AsyncDashboardView" -->
                <!-- <KeepAlive
                    v-else-if="myStore.dashboardsLayoutMode === 'replace'"
                >
                    <CompDashboardView
                        :key="currentDashboardId"
                        :compDbId="currentDashboardId"
                    />
                </KeepAlive> -->
                <template v-else>
                    <!-- AsyncDashboardView -->
                    <CompDashboardView
                        v-for="d in myStore.compDashboardList"
                        v-show="d.id === currentDashboardId || d.isRepresented"
                        :key="d.id"
                        :compDbId="d.id"
                    />
                </template>
            </div>
        </el-scrollbar>
        <div class="footer">
            <div class="recent-bar">
                <recentSvg style="width: 70%; height: 70%" />
                <template
                    v-for="(d, i) in renderableRecentCompDashboards"
                    :key="d.id"
                >
                    <el-popover
                        placement="top"
                        trigger="hover"
                        :width="220"
                        @after-enter="
                            async () => {
                                if (d.id === currentDashboardId)
                                    await myStore.calcPrincipalSnapshotOfDashboard(
                                        d.id
                                    );
                            }
                        "
                    >
                        <template #reference>
                            <div
                                class="recent-item"
                                :class="{
                                    active: d.id === currentDashboardId,
                                }"
                                @click="(e) => handleRecentClick(e, d.id)"
                            >
                                <dashboardRefSvg class="recent-item-icon" />
                                <div>
                                    {{
                                        myStore.getCompDashboardById(d.id)?.name
                                    }}
                                </div>
                            </div>
                        </template>
                        <div style="text-align: start">
                            <h4>Principal View Snapshot</h4>
                            <el-divider />
                            <div
                                :style="{
                                    margin: '0 auto',
                                    width: '200px',
                                    border: '1px solid black',
                                    boxSizing: 'content-box',
                                    height:
                                        ((principalViews[i]?.bodyHeight ||
                                            200) *
                                            200) /
                                            (principalViews[i]?.bodyWidth ||
                                                200) +
                                        2 +
                                        'px',
                                }"
                            >
                                <LoadingComp
                                    v-if="principalViews[i]?.isGettingSnapshot"
                                    :text="`loading snapshot of view ${principalViews[i]?.viewName}`"
                                />
                                <ErrorComp
                                    v-else-if="
                                        principalViews[i]?.gettingSnapShotError
                                    "
                                    :error="
                                        principalViews[i]?.gettingSnapShotError
                                    "
                                />
                                <img
                                    v-else-if="
                                        principalViews[i]?.snapshotBase64
                                    "
                                    :style="{
                                        width: '200px',
                                        cursor: 'pointer',
                                    }"
                                    :src="principalViews[i]?.snapshotBase64"
                                    :alt="`the snapshot of view ${principalViews[i]?.viewName}`"
                                    @click="(e) => handleRecentClick(e, d.id)"
                                />
                            </div>
                        </div>
                    </el-popover>
                </template>
            </div>

            <el-divider direction="vertical"></el-divider>

            <el-popover
                placement="top-end"
                trigger="hover"
                :width="1200"
                @after-enter="handleTreeOpen"
                @after-leave="
                    () => {
                        isShowingTree = false;
                    }
                "
            >
                <template #reference>
                    <treeSvg
                        style="
                            width: 2em;
                            height: 2em;
                            color: var(--el-text-color-primary);
                        "
                    />
                </template>
                <div style="text-align: start">
                    <h3>Dashboard History Tree View</h3>
                    <el-divider />

                    <TreeCalc
                        v-if="isShowingTree"
                        :width="1150"
                        :height="400"
                        :is-single="false"
                    />
                </div>
            </el-popover>
        </div>
    </div>
</template>

<script setup lang="ts">
import { storeToRefs } from "pinia";
import { useRoute } from "vue-router";
import { useMyStore } from "@/stores/store";
import { useWebWorkerFn } from "@/utils/myWebWorker";
import {
    clearSelModes,
    whenToRescaleModes,
    settingsMenuTriggerModes,
    dashboardsLayoutModes,
} from "@/stores/enums";
import {
    defineAsyncComponent,
    onUnmounted,
    watch,
    ref,
    computed,
    toRaw,
} from "vue";
import { baseUrl } from "../api/api";
import {
    calcGraphCoords,
    calcSeparatedMultiGraphCoords,
    calcTsne,
} from "../utils/graphUtils";

import { default as setting } from "@/components/icon/FluentSettings48Regular.vue";
import { default as recentSvg } from "../components/icon/CarbonRecentlyViewed.vue";
import { default as treeSvg } from "../components/icon/CarbonDecisionTree.vue";
import { default as dashboardRefSvg } from "../components/icon/CarbonDashboardReference.vue";
import LoadingComp from "../components/state/Loading.vue";
import ErrorComp from "../components/state/Error.vue";
import CompDashboardView from "../components/compDashboardView.vue";
import TreeCalc from "../components/footer/tree.vue";

import { nanoid } from "nanoid";
import type {
    Dataset,
    CompDashboard,
    TsneCoord,
    Type_TaskTypes,
} from "@/types/types";

const h1Ref = ref<HTMLDivElement | null>(null);
const h1Width = computed(() => h1Ref.value?.clientWidth + "px");

const route = useRoute();
const myStore = useMyStore();
const isLoading = ref(true);
const loadingText = ref<string>("");
const loadingError = ref<Error>();

const { datasetName1, datasetName2 } = route.params;
// console.log("in comp layout, taskType: ", taskType);
const currentDsTaskType = computed(() => {
    let taskType: Type_TaskTypes | undefined = undefined;
    try {
        const ds1 = myStore.getDatasetByName(datasetName1 as string);
        const ds2 = myStore.getDatasetByName(datasetName2 as string);
        taskType = ds1.taskType || ds2.taskType || "";
    } catch (e) {
        //
    }
    return taskType;
});

const load2Datasets = async () => {
    let ret1, ret2;
    if (
        //赋值+判断。
        !(ret1 = myStore.datasetList.find(
            (d) => d.name === datasetName1 && d.isComplete
        ))
    ) {
        ret1 = await myStore.fetchOriginDataset(
            baseUrl,
            datasetName1 as string
        );
        ret1.isComplete = true; // 确认完成
        myStore.addDataset(ret1 as Dataset); //放在这的好处：如果放在fetchOrigin，这里会再取一次
    }

    if (
        !(ret2 = myStore.datasetList.find(
            (d) => d.name === datasetName2 && d.isComplete
        ))
    ) {
        ret2 = await myStore.fetchOriginDataset(
            baseUrl,
            datasetName2 as string,
            false,
            false
        ); //save ground truths only once

        ret2.isComplete = false;
        myStore.addDataset(ret2 as Dataset);
    }
    // ret1.taskType = "111";//DEBUG
    // check if two datasets are compatible
    if (ret1!.taskType !== ret2!.taskType) {
        throw new Error("Two datasets have different task types!");
    }
    if (Object.hasOwn(ret1!, "embNode") !== Object.hasOwn(ret2!, "embNode")) {
        throw new Error("One dataset has embNode but the other does not!");
    } else if (
        Object.hasOwn(ret1!, "predLabels") !==
        Object.hasOwn(ret2!, "predLabels")
    ) {
        throw new Error("One dataset has predLabels but the other does not!");
    } else if (
        Object.hasOwn(ret1!, "embNode") &&
        Object.hasOwn(ret2!, "embNode") &&
        ret1!.embNode!.length !== ret2!.embNode!.length
    ) {
        throw new Error("Two datasets have different lengths on embNodes!");
    } else if (
        Object.hasOwn(ret1!, "predLabels") &&
        Object.hasOwn(ret2!, "predLabels") &&
        ret1!.predLabels!.length !== ret2!.predLabels!.length
    ) {
        throw new Error("Two datasets have different lengths on predLabels!");
    } else {
        console.log("in load2Datasets, passed check!");
    }
};

const calcFirstDashboard = async (): Promise<Partial<CompDashboard>> => {
    const ds1 = myStore.getDatasetByName(datasetName1 as string) as Dataset;
    const ds2 = myStore.getDatasetByName(datasetName2 as string) as Dataset;
    // console.log("in calcFirstDb, ds1, ds2", ds1, ds2);

    /////////////////////////////////////////
    ////// calc force-directed layout
    let graphCoordsRet;
    if (ds1.taskType !== "graph-classification") {
        const {
            workerFn: graphWorkerFn,
            // workerStatus: graphWorkerStatus,
            workerTerminate: graphWorkerTerminate,
        } = useWebWorkerFn(calcGraphCoords, {
            timeout: 20_000,
            dependencies: [
                // "http://localhost:5173/workers/d3js.org_d3.v7.js",
                // "https://d3js.org/d3.v7.js", //for debug
                "https://d3js.org/d3.v7.min.js",
            ],
        });
        if (ds1.graphCoordsRet) {
            graphCoordsRet = ds1.graphCoordsRet; // REVIEW shallow copy
        } else if (ds2.graphCoordsRet) {
            graphCoordsRet = ds2.graphCoordsRet; // REVIEW shallow copy
        } else {
            graphCoordsRet = await graphWorkerFn(
                toRaw(ds1.nodes || ds2.nodes || []),
                toRaw(ds1.links || ds2.links || [])
            );
            console.log(
                "in calcGraphCoords, useWebWorkerFn, got ret:",
                graphCoordsRet
            );
            graphWorkerTerminate();
        }
    } else {
        if (ds1.graphCoordsRet) {
            graphCoordsRet = ds1.graphCoordsRet;
        } else if (ds2.graphCoordsRet) {
            graphCoordsRet = ds2.graphCoordsRet;
        } else {
            const {
                workerFn: graphWorkerFn,
                // workerStatus: graphWorkerStatus,
                workerTerminate: graphWorkerTerminate,
            } = useWebWorkerFn(calcSeparatedMultiGraphCoords, {
                timeout: 40_000,
                dependencies: [
                    // "https://d3js.org/d3.v7.js",
                    "https://d3js.org/d3.v7.min.js",
                    "http://localhost:5173/workers/forceLayout.js",
                ],
            });
            const graphArrForCalc = toRaw(ds1.graphArr || ds2.graphArr).map(
                (g) => {
                    return {
                        ...g,
                        links: g.links.map((linkId) =>
                            toRaw(
                                ds1.globalLinksRichDict[linkId] ||
                                    ds2.globalLinksRichDict[linkId]
                            )
                        ),
                    };
                }
            );
            console.log(
                "in calcSeparatedMultiGraphCoords, useWebWorkerFn, ready to calc, parma graphData is,",
                graphArrForCalc,
                "\nds.globalLinksRichDict is,",
                ds1.globalLinksRichDict
            );
            graphCoordsRet = await graphWorkerFn(
                graphArrForCalc,
                ds1?.graphArr.length || ds2?.graphArr.length
                // () => 100,
                // () => 100,
                // undefined,
                // 2
            );
            console.log(
                "in calcSeparatedMultiGraphCoords, useWebWorkerFn, got ret:",
                graphCoordsRet
            );
            graphWorkerTerminate();
        }
    }
    ////// calc force-directed layout
    /////////////////////////////////////////

    /////////////////////////////////////////
    ///// calc dimension reduction

    let tsneRet1, tsneRet2;
    if (ds1.tsneRet) {
        tsneRet1 = ds1.tsneRet; // REVIEW shallow copy
    } else {
        const {
            workerFn: tsneWorkerFn1,
            // workerStatus: tsneWorkerStatus,
            workerTerminate: tsneWorkerTerminate1,
        } = useWebWorkerFn(calcTsne, {
            timeout: 60_000,
            dependencies: ["http://localhost:5173/workers/tsne.js"],
        });

        tsneRet1 = (await tsneWorkerFn1(toRaw(ds1.embNode))).map((d, i) => ({
            id: i + "",
            x: d[0],
            y: d[1],
        })) as Array<TsneCoord>;
        console.log("in calcTsne, useWebWorkerFn, got ret", tsneRet1);
        tsneWorkerTerminate1();
    }

    if (ds2.tsneRet) {
        tsneRet2 = ds2.tsneRet; // REVIEW shallow copy
    } else {
        const {
            workerFn: tsneWorkerFn2,
            // workerStatus: tsneWorkerStatus,
            workerTerminate: tsneWorkerTerminate2,
        } = useWebWorkerFn(calcTsne, {
            timeout: 60_000,
            dependencies: ["http://localhost:5173/workers/tsne.js"],
        }); //NOTE 需要定义两个worker，因为如果是同一个，后一个会终止前一个
        tsneRet2 = (await tsneWorkerFn2(toRaw(ds2.embNode))).map((d, i) => ({
            id: i + "",
            x: d[0],
            y: d[1],
        })) as Array<TsneCoord>;

        console.log("in calcTsne, useWebWorkerFn, got ret", tsneRet2);
        tsneWorkerTerminate2();
    }
    ///// calc dimension reduction
    /////////////////////////////////////////

    /////////////////////////////////////////
    ///// calc graph dimension reduction
    let graphTsneRet1, graphTsneRet2;
    if (ds1.taskType === "graph-classification") {
        const {
            workerFn: graphTsneWorkerFn1,
            // workerStatus: graphTsneWorkerStatus1,
            workerTerminate: graphTsneWorkerTerminate1,
        } = useWebWorkerFn(calcTsne, {
            timeout: 60_000,
            dependencies: ["http://localhost:5173/workers/tsne.js"],
        });

        const {
            workerFn: graphTsneWorkerFn2,
            // workerStatus: graphTsneWorkerStatus2,
            workerTerminate: graphTsneWorkerTerminate2,
        } = useWebWorkerFn(calcTsne, {
            timeout: 60_000,
            dependencies: ["http://localhost:5173/workers/tsne.js"],
        });
        if (ds1.graphTsneRet) {
            graphTsneRet1 = ds1.graphTsneRet; // REVIEW shallow copy
        } else {
            graphTsneRet1 = (await graphTsneWorkerFn1(toRaw(ds1.embNode))).map(
                (d, i) => ({
                    id: i + "",
                    x: d[0],
                    y: d[1],
                })
            ) as Array<TsneCoord>;
            console.log(
                "in calc graph tsne 1, useWebWorkerFn, got ret",
                graphTsneRet1
            );
            graphTsneWorkerTerminate1();
        }
        if (ds2.graphTsneRet) {
            graphTsneRet2 = ds2.graphTsneRet; // REVIEW shallow copy
        } else {
            graphTsneRet2 = (await graphTsneWorkerFn2(toRaw(ds2.embNode))).map(
                (d, i) => ({
                    id: i + "",
                    x: d[0],
                    y: d[1],
                })
            ) as Array<TsneCoord>;
            console.log(
                "in calc graph tsne 2, useWebWorkerFn, got ret",
                graphTsneRet2
            );
            graphTsneWorkerTerminate2();
        }
    }
    ///// calc graph dimension reduction
    /////////////////////////////////////////
    return {
        //ANCHOR DEFINITION related
        refDatasetsNames: [datasetName1 as string, datasetName2 as string],
        parentId: "",
        fromViewName: "", //从上一个的db 的哪个view选出来的。
        isRoot: true,
        srcNodesDict: ds1.globalNodesDict || ds2.globalNodesDict!,
        graphCoordsRet,
        srcLinksArr: ds1.links || ds2.links || [],
        srcLinksDict: ds1.globalLinksDict || ds2.globalLinksDict!,

        tsneRet1,
        tsneRet2,
        graphTsneRet1,
        graphTsneRet2,
    };
};

const addFirstDashboard = async () => {
    // myStore

    const ds1 = myStore.getDatasetByName(datasetName1 as string) as Dataset;
    const ds2 = myStore.getDatasetByName(datasetName2 as string) as Dataset;
    let id;
    if (
        !(
            myStore.compDashboardList.find(
                (d) =>
                    d.isRoot === true &&
                    d.refDatasetsNames[0] === datasetName1 &&
                    d.refDatasetsNames[1] === datasetName2
            )?.isComplete === true
        )
    ) {
        const ret = await calcFirstDashboard();
        id = nanoid();
        myStore.addCompDashboard(
            {
                ...ret,
                id,
                isComplete: true,
                date: Date.now ? Date.now() : new Date().getTime(),
                fromViewName: myStore.defaultCompViewNames[ds1.taskType][0],
            },
            myStore.defaultCompViewNames[ds1.taskType]
        );
    }
};

// watch(
//     [() => route.params.datasetName1, () => route.params.datasetName2],
(() => {
    if (myStore.layoutLoadPromise)
        myStore.layoutLoadPromiseReject("new comp layout encountered!");
    myStore.layoutLoadPromise = new Promise<void>((resolve, reject) => {
        myStore.layoutLoadPromiseReject = reject;
        isLoading.value = true;
        loadingError.value = undefined;
        loadingText.value = "load 2 datasets";
        load2Datasets()
            .then(() => {
                loadingText.value = "add first dashboard";
                addFirstDashboard()
                    .then(() => {
                        console.log(
                            "in comp Layout, after add firstDb, now store recentList",
                            myStore.recentCompDashboardList
                        );
                        console.log(
                            "in comp Layout, after add firstDb, now currentID",
                            currentDashboardId.value
                        );
                        resolve();
                        isLoading.value = false;
                    })
                    .catch((error) => {
                        reject(error);
                        isLoading.value = false;
                        loadingError.value = error;
                    });
            })
            .catch((error) => {
                console.error(
                    "myStore.layoutLoadPromise, in load2Datasets, caught error:",
                    error
                );
                reject(error);
                isLoading.value = false;
                loadingError.value = error;
            });
    });
})();
//     { immediate: true }
// );

// const isLoading = ref(true);
// const error = ref();
// const loadData = async () => {
//     try {
//         isLoading.value = true;
//         error.value = null;
//         await myStore.compLayoutLoadPromise;
//         console.log("in compare layout, loading is going to finish");
//     } catch (e) {
//         error.value = e;
//     } finally {
//         isLoading.value = false;
//     }
// };
// loadData();

// const AsyncDashboardView = defineAsyncComponent({
//     loader: async () => {
//         console.log("in comp Layout, in define Async CompDashboard, loader");
//         await myStore.layoutLoadPromise;
//         return await import("../components/compDashboardView.vue");
//     },
//     delay: 200,
//     timeout: 50_000,
//     loadingComponent: LoadingComp,
//     errorComponent: ErrorComp,
// });

const { renderableRecentCompDashboards } = storeToRefs(myStore);
const currentDashboardId = computed(
    () => myStore.recentCompDashboardList.at(-1)?.id || "????????????????"
);
const principalViews = computed(() =>
    renderableRecentCompDashboards.value.map((d) =>
        myStore.getPrincipalViewOfDashboard(d.id)
    )
);
const handleRecentClick = async (e: Event, id: string) => {
    await myStore.calcPrincipalSnapshotOfDashboard(currentDashboardId.value);
    myStore.toCompDashboardById(id);
};

const isShowingTree = ref(false);
// when toggle isAppend or isReplace, give a pop confirm

const handleTreeOpen = async () => {
    //NOTE  we choose to calc snapshot when `<popover>` open,
    await myStore.calcPrincipalSnapshotOfDashboard(currentDashboardId.value);
    isShowingTree.value = true;
};

const resolveIsAppend = ref<null | ((state: boolean) => void)>(null);
const rejectIsAppend = ref<null | ((reason: string) => void)>(null);
const isAppendChangePromise = ref<Promise<boolean> | null>(null);
const beforeIsAppendChange = async () => {
    if (
        isAppendChangePromise.value &&
        isAppendChangePromise.value instanceof Promise
    ) {
        // console.log("a former exist!");
        rejectIsAppend.value!("cancel former unsettled confirm");
        rejectIsAppend.value = null;
    }
    isAppendChangePromise.value = new Promise<boolean>((resolve, reject) => {
        resolveIsAppend.value = resolve;
        rejectIsAppend.value = reject;
    });
    return isAppendChangePromise.value;
};
const confirmIsAppend = async () => {
    resolveIsAppend.value!(true);
    isAppendChangePromise.value = null;
};
const cancelIsAppend = async () => {
    rejectIsAppend.value!("user canceled switch");
    isAppendChangePromise.value = null;
};

onUnmounted(() => {
    myStore.layoutLoadPromiseReject("compareLayout unMounted!");
    myStore.datasetList = [];
    myStore.compDashboardList = [];
    myStore.layoutLoadPromise = null;
});
</script>

<style scoped>
.container {
    display: flex;
    flex-direction: column;
    height: 100vh;
}
.header-flex,
.footer {
    width: calc(100% - 40px);
    height: 5vh;
    padding: 0 20px;
    background-color: rgb(102, 177, 255);
    color: var(--el-text-color-primary);

    flex: 0 0 auto;

    display: flex;
    justify-content: space-between;
    align-items: center;
}
.header-all h1 {
    position: absolute;
    margin: 0;
    padding: 0;
    top: 0px;
    left: calc((100% - v-bind("h1Width")) / 2);
}
/* .footer {
    position: fixed;
    top: calc(100% - 60px);
    left: 0;
} */

.main {
    flex: 1 1 auto;
    padding: 20px;
    position: relative;
    /* border: 2px solid black; */
    /* height: calc(100vh - 120px); */ /*会导致main的padding-bottom无效*/
    /* min-height: 1080px; */
}

.tree-card {
    width: 1200px;
    height: 500px;
    overflow-y: scroll;
    overflow-x: scroll;
}
.setting-item-name {
    font-weight: bold;
}
.recent-item {
    background-color: aliceblue;
    width: 130px;
    font-size: 10px;
    height: 70%;
    border-radius: 4px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.12), 0 0 6px rgba(0, 0, 0, 0.04);
    cursor: pointer;

    display: grid;
    grid-template-columns: 1fr 1fr;
    align-items: center;
    justify-items: center;
}
.recent-item-icon {
    width: 20px;
    height: 20px;
}
.active {
    border: 2.4px black solid;
    background-color: rgba(184, 215, 255);
}
.recent-bar {
    display: grid;
    grid-template-columns: 1fr repeat(
            v-bind("renderableRecentCompDashboards.length"),
            2fr
        );
    align-items: center;
    justify-items: stretch;
    column-gap: 0.5em;
    height: 100%;
}
</style>
