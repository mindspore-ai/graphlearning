<template>
    <div class="container">
        <div class="header-all">
            <h1 ref="h1Ref">CorGIE-2 Single</h1>
            <div class="header-flex">
                <div>
                    <RouterLink to="/home" :replace="true">
                        <el-button>Back to Home</el-button></RouterLink
                    >
                    <el-divider direction="vertical"></el-divider>

                    <el-popover
                        placement="bottom-start"
                        trigger="click"
                        :width="400"
                    >
                        <template #reference>
                            <el-button @click="handleCompareClick"
                                >Add Comparison</el-button
                            >
                        </template>
                        <div>
                            <h3>Select a dataset to compare</h3>
                            <el-divider />
                            <div
                                v-loading="isUrlLoading"
                                style="text-align: start; min-height: 250px"
                            >
                                <div v-if="compareList.length > 0">
                                    <RouterLink
                                        v-for="d in compareList"
                                        :key="d.name"
                                        :to="`/compare/${datasetName}/${d.name}`"
                                        style="
                                            display: block;
                                            text-decoration: none;
                                        "
                                        :replace="true"
                                        >{{ d.name }}</RouterLink
                                    >
                                </div>
                                <div v-else>No comparable datasets</div>
                            </div>
                        </div>
                    </el-popover>
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
                <!-- <Transition
                    name="dashboard-transition"
                    :css="true"
                    mode="out-in"
                    v-else-if="myStore.dashboardsLayoutMode === 'replace'"
                > -->
                <!-- @after-leave="onAfterLeave" -->

                <!-- <KeepAlive
                    v-else-if="myStore.dashboardsLayoutMode === 'replace'"
                >
                    <component
                        :is="SingleDashboardView"
                        :key="currentDashboardId"
                        :singleDbId="currentDashboardId"
                    />
                </KeepAlive> -->

                <!-- </Transition> -->
                <!-- <TransitionGroup name="list"> -->
                <!-- </TransitionGroup> -->
                <!--如果使用TransitionGroup, bug非常难修复。-->
                <template v-else>
                    <!-- <KeepAlive> -->
                    <SingleDashboardView
                        v-for="d in myStore.singleDashboardList"
                        v-show="d.id === currentDashboardId || d.isRepresented"
                        :key="d.id"
                        :singleDbId="d.id"
                    />
                    <!-- </KeepAlive > -->
                </template>
            </div>
        </el-scrollbar>

        <div class="footer">
            <div class="recent-bar">
                <recentSvg style="width: 70%; height: 70%" />
                <template
                    v-for="(d, i) in renderableRecentSingleDashboards"
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
                                        myStore.getSingleDashboardById(d.id)
                                            ?.name
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
                trigger="click"
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
                    />
                </div>
            </el-popover>
        </div>
    </div>
</template>

<script setup lang="ts">
import {
    ref,
    computed,
    watch,
    nextTick,
    toRaw,
    defineAsyncComponent,
    onUnmounted,
} from "vue";
import { storeToRefs } from "pinia";
import { useRoute } from "vue-router";
import { baseUrl } from "../api/api";
import { useMyStore } from "@/stores/store";
import {
    clearSelModes,
    whenToRescaleModes,
    settingsMenuTriggerModes,
    dashboardsLayoutModes,
    taskTypes,
} from "@/stores/enums";
import {
    type Type_TaskTypes,
    type SingleDashboard,
    type Dataset,
    type UrlData,
    type Node,
    type Link,
    type TsneCoord,
    type View,
} from "@/types/types";
import { useWebWorkerFn } from "@/utils/myWebWorker";
import {
    calcGraphCoords,
    calcSeparatedMultiGraphCoords,
    calcTsne,
} from "@/utils/graphUtils";
import { default as setting } from "@/components/icon/FluentSettings48Regular.vue";
import type { ScrollbarInstance } from "element-plus";
import { default as recentSvg } from "../components/icon/CarbonRecentlyViewed.vue";
import { default as treeSvg } from "../components/icon/CarbonDecisionTree.vue";
import { default as dashboardRefSvg } from "../components/icon/CarbonDashboardReference.vue";
import LoadingComp from "../components/state/Loading.vue";
import TreeCalc from "../components/footer/tree.vue";
import ErrorComp from "../components/state/Error.vue";
import { customAlphabet } from "nanoid";

import SingleDashboardView from "@/components/singleDashboardView.vue";

const h1Ref = ref<HTMLDivElement | null>(null);
const h1Width = computed(() => h1Ref.value?.clientWidth + "px");

const route = useRoute();
const myStore = useMyStore();
const isLoading = ref(true);
const loadingText = ref<string>("");
const loadingError = ref<Error>();

const { datasetName } = route.params;
const currentDsTaskType = computed(() => {
    let taskType: Type_TaskTypes | undefined = undefined;
    try {
        const ds = myStore.getDatasetByName(datasetName as string);
        taskType = ds.taskType || "";
    } catch (e) {
        //
    }
    return taskType;
});
const loadDataset = async () => {
    if (
        !myStore.datasetList.some((d) => d.name === datasetName && d.isComplete)
    ) {
        const ret = await myStore.fetchOriginDataset(
            baseUrl,
            datasetName as string
        );
        ret.isComplete = true; // 确认完成。
        myStore.addDataset(ret as Dataset); //放在这的好处：如果放在fetchOrigin，这里会再取一次
    }
};
const calcFirstDashboard = async (): Promise<Partial<SingleDashboard>> => {
    const ds = myStore.getDatasetByName(datasetName as string)!;
    // console.log("in singleLayout calcFirstDb, ds1, ds2", ds1, ds2);

    /////////////////////////////////////////
    ////// calc force-directed layout
    let graphCoordsRet;
    if (ds.taskType !== "graph-classification") {
        const {
            workerFn: graphWorkerFn,
            // workerStatus: graphWorkerStatus,
            workerTerminate: graphWorkerTerminate,
        } = useWebWorkerFn(calcGraphCoords, {
            timeout: 20_000,
            dependencies: [
                "http://localhost:5173/workers/d3js.org_d3.v7.js",
                // "https://d3js.org/d3.v7.js", //for debug
                // "https://d3js.org/d3.v7.min.js"
            ],
        });
        if (ds.graphCoordsRet) {
            graphCoordsRet = ds.graphCoordsRet; // REVIEW shallow copy
        } else {
            graphCoordsRet = await graphWorkerFn(
                toRaw(ds.nodes!),
                toRaw(ds.links!)
            );
            console.log(
                "in calcGraphCoords, useWebWorkerFn, got ret:",
                graphCoordsRet
            );
            graphWorkerTerminate();
        }
    } else {
        const {
            workerFn: graphWorkerFn,
            // workerStatus: graphWorkerStatus,
            workerTerminate: graphWorkerTerminate,
        } = useWebWorkerFn(calcSeparatedMultiGraphCoords, {
            timeout: 40_000,
            dependencies: [
                "https://d3js.org/d3.v7.js",
                "http://localhost:5173/workers/forceLayout.js",
            ],
        });
        if (ds.graphCoordsRet) {
            graphCoordsRet = ds.graphCoordsRet;
        } else {
            const graphArrForCalc = toRaw(ds.graphArr).map((g) => {
                return {
                    ...g,
                    links: g.links.map((linkId) =>
                        toRaw(ds.globalLinksRichDict[linkId])
                    ),
                };
            });
            console.log(
                "in calcSeparatedMultiGraphCoords, useWebWorkerFn, ready to calc, parma graphData is,",
                graphArrForCalc,
                "\nds.globalLinksRichDict is,",
                ds.globalLinksRichDict
            );
            graphCoordsRet = await graphWorkerFn(
                graphArrForCalc,
                ds.graphArr.length
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
    const {
        workerFn: tsneWorkerFn,
        // workerStatus: tsneWorkerStatus,
        workerTerminate: tsneWorkerTerminate,
    } = useWebWorkerFn(calcTsne, {
        timeout: 60_000,
        dependencies: ["http://localhost:5173/workers/tsne.js"],
    });

    let tsneRet;
    if (ds.tsneRet) {
        tsneRet = ds.tsneRet; // REVIEW shallow copy
    } else {
        tsneRet = (await tsneWorkerFn(toRaw(ds.embNode))).map((d, i) => ({
            id: i + "",
            x: d[0],
            y: d[1],
        })) as Array<TsneCoord>;
        console.log("in calcTsne, useWebWorkerFn, got ret", tsneRet);
        tsneWorkerTerminate();
    }
    ///// calc dimension reduction
    /////////////////////////////////////////

    /////////////////////////////////////////
    ///// calc graph dimension reduction
    let graphTsneRet;
    if (ds.taskType === "graph-classification") {
        const {
            workerFn: graphTsneWorkerFn,
            // workerStatus: graphTsneWorkerStatus,
            workerTerminate: graphTsneWorkerTerminate,
        } = useWebWorkerFn(calcTsne, {
            timeout: 60_000,
            dependencies: ["http://localhost:5173/workers/tsne.js"],
        });

        if (ds.graphTsneRet) {
            graphTsneRet = ds.graphTsneRet; // REVIEW shallow copy
        } else {
            graphTsneRet = (await graphTsneWorkerFn(toRaw(ds.embNode))).map(
                (d, i) => ({
                    id: i + "",
                    x: d[0],
                    y: d[1],
                })
            ) as Array<TsneCoord>;
            console.log(
                "in calc graph tsne, useWebWorkerFn, got ret",
                graphTsneRet
            );
            graphTsneWorkerTerminate();
        }
    }
    ///// calc graph dimension reduction
    /////////////////////////////////////////

    return {
        //ANCHOR DEFINITION related
        refDatasetName: datasetName as string,
        parentId: "",
        fromViewName: "", //从上一个的db 的哪个view选出来的。
        isRoot: true,
        srcNodesDict: ds.globalNodesDict,
        graphCoordsRet,
        srcLinksArr: ds.links || [],
        srcLinksDict: ds.globalLinksDict,
        tsneRet,
        graphTsneRet,
    };
};

const addFirstDashboard = async () => {
    // myStore
    const ds = myStore.getDatasetByName(datasetName as string)!;
    let id;
    if (
        !(
            myStore.singleDashboardList.find(
                (d) => d.isRoot === true && d.refDatasetName === datasetName
            )?.isComplete === true
        )
    ) {
        const ret = await calcFirstDashboard();
        const nanoid = customAlphabet("abcdefghijklmnopqrstuvwxyz_", 10);
        id = nanoid(); //=> "不含数字"
        myStore.addSingleDashboard(
            {
                ...ret,
                id,
                isComplete: true,
                date: Date.now ? Date.now() : new Date().getTime(),
                fromViewName: myStore.defaultSingleViewNames[ds.taskType][0],
            },
            myStore.defaultSingleViewNames[ds.taskType]
        );
    }
};

const currentDashboardId = computed(
    () => myStore.recentSingleDashboardList.at(-1)?.id || "?????????"
);

watch(
    [() => route.params.datasetName1],
    () => {
        if (myStore.layoutLoadPromise)
            myStore.layoutLoadPromiseReject("new comp layout encountered!");
        myStore.layoutLoadPromise = new Promise<void>((resolve, reject) => {
            isLoading.value = true;
            myStore.layoutLoadPromiseReject = reject;
            loadingError.value = undefined;
            loadingText.value = "load dataset";
            loadDataset()
                .then(() => {
                    loadingText.value = "add first dashboard";
                    addFirstDashboard()
                        .then(() => {
                            console.log(
                                "in single Layout, after add firstDb, now store recentList",
                                myStore.recentSingleDashboardList
                            );
                            // loadingText.value = "setting current dbId";
                            // currentDashboardId.value =
                            //     myStore.recentSingleDashboardList.at(-1)?.id ||
                            //     "";
                            console.log(
                                "in single Layout, after add firstDb, now currentID",
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
                        "myStore.layoutLoadPromise, in loadDataset, caught error:",
                        error
                    );
                    reject(error);
                    isLoading.value = false;
                    loadingError.value = error;
                });
        });
    },
    { immediate: true }
);

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION history related
const { renderableRecentSingleDashboards } = storeToRefs(myStore);
const principalViews = computed(() =>
    renderableRecentSingleDashboards.value.map((d) =>
        myStore.getPrincipalViewOfDashboard(d.id)
    )
);
const handleRecentClick = async (e: Event, id: string) => {
    await myStore.calcPrincipalSnapshotOfDashboard(currentDashboardId.value);
    myStore.toSingleDashboardById(id);
};

const isShowingTree = ref(false);

const handleTreeOpen = async () => {
    //NOTE  we choose to calc snapshot when `<popover>` open,
    await myStore.calcPrincipalSnapshotOfDashboard(currentDashboardId.value);
    isShowingTree.value = true;
};
////////////////// !SECTION history related
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION add comparison related
const isUrlLoading = ref(true);
const urlError = ref<Error>();
const compareList = ref<Array<UrlData>>([]);
const handleCompareClick = async (e: Event) => {
    if (myStore.urlList && myStore.urlList.length) {
        const { graph, name } = myStore.urlList.find(
            (d) => d.name === datasetName
        )!;
        compareList.value =
            myStore.urlList.filter(
                (d) => d.graph === graph && d.name !== datasetName
            ) || [];
        isUrlLoading.value = false;
    } else {
        try {
            isUrlLoading.value = true;
            urlError.value = undefined;
            const res = await new Promise<Response>((resolve, reject) => {
                setTimeout(() => resolve(fetch(baseUrl + "list.json")), 1000);
            }); //mock a delay
            // const res = await fetch(baseUrl + "list.json");
            if (res.ok) {
                const dataJson = await res.json();
                console.log("in add Compare: dataJson :>> ", dataJson);
                const { graph, model } = dataJson.list.find(
                    (d: UrlData) => d.name === datasetName
                );
                compareList.value = dataJson.list.filter(
                    (d: UrlData) => d.graph === graph && d.name !== datasetName
                );
            } else throw new Error("get comparative list failed!");
        } catch (e: any) {
            console.warn(e);
            urlError.value = e;
        } finally {
            isUrlLoading.value = false;
        }
    }
    myStore.repairButtonFocus(e);
};
////////////////// !SECTION add comparison related
////////////////////////////////////////////////////////////////////////////////

const scrollRef = ref<ScrollbarInstance | null>(null);
//当按下按钮，toDashboardById调用，currentDashboardId也会变化, 从而接着引发此函数的变化
//这样的好处是将此函数的变化和按钮解耦，使得append一个新dashboard时，也能滚动。
watch(currentDashboardId, async (newV, oldV) => {
    console.log("in single Layout: current dbId changed!,", oldV, newV);
    if (myStore.dashboardsLayoutMode === "append") {
        await nextTick();
        const childComp = document.getElementById(newV)!;
        scrollRef.value?.setScrollTop(childComp.offsetTop!);
    }
});

// when toggle isAppend or isReplace, give a pop confirm
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
    myStore.layoutLoadPromiseReject("singleLayout unMounted!");
    myStore.datasetList = [];
    myStore.singleDashboardList = [];
    myStore.layoutLoadPromise = null;
});
</script>

<style scoped>
.list-move, /* 对移动中的元素应用的过渡 */
.list-enter-active,
.list-leave-active {
    transition: all 1s ease;
}

.list-enter-from,
.list-leave-to {
    position: absolute;
    left: 20px;
    top: 100%;
    width: 60px;
    height: 30px;
    opacity: 0;
}

/* 确保将离开的元素从布局流中删除
  以便能够正确地计算移动的动画。 */
.list-leave-active {
    position: absolute;
}

/********************************** */

.dashboard-transition-enter-active {
    transition: all 0.3s ease-out;
}
.dashboard-transition-leave-active {
    transition: all 0.3s ease-in;
}

.dashboard-transition-enter-to,
.dashboard-transition-leave-from {
    /*在固定下不是用定位做的(，但是动画的起始点可以用于position*/
    position: absolute;
    left: 20px;
    top: 20px;
    width: calc(100% - 40px);
}
.dashboard-transition-enter-from,
.dashboard-transition-leave-to {
    position: absolute;
    left: 20px;
    top: 100vh;
    width: 20px;
    height: 20px;

    opacity: 0;
}

/* .dashboard-transition-leave-active .dashboard-child,
.dashboard-transition-enter-active .dashboard-child {
    transition: all 2s ease-in-out;
    transition-delay: 3s;
}

.dashboard-transition-enter-from .dashboard-child,
.dashboard-transition-leave-to .dashboard-child {
    opacity: 0;
} */

/************************************************* */

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
            v-bind("renderableRecentSingleDashboards.length"),
            2fr
        );
    align-items: center;
    justify-items: stretch;
    column-gap: 0.5em;
    height: 100%;
}
/* .recent-bar > * {
    display: inline-block;
}
.recent-bar > :not(:last-child) {
    margin-right: 50px;
} */
.loading {
    text-align: center;
}

.el-container,
.container {
    /* width: 100%; */
    /* height: 100%; */

    display: flex;
    flex-direction: column;
    height: 100vh;
}

/* .el-aside {
    box-sizing: content-box;
    border-right: rgb(102, 177, 255) 1px solid;
} */

.el-main,
.main {
    flex: 1 1 auto;
    padding: 20px;
    position: relative;
    /* min-height: 960px; */
}
.el-header,
.el-footer,
.header-flex,
.footer {
    /* width: 100%; */
    /* height: 60px; */

    width: calc(100% - 40px);
    height: 5vh;
    padding: 0 20px;
    background-color: rgb(102, 177, 255);
    color: var(--el-text-color-primary);

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
</style>
