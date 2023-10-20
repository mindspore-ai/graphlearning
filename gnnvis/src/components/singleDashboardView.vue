<template>
    <div
        :id="`single-dashboard-${props.singleDbId}`"
        :style="style"
        :class="{
            'single-dashboard': !db.isRepresented,

            'represented-single-dashboard': db.isRepresented,
        }"
        :ref="db.isRepresented ? 'representedDashboardRef' : 'dashboardRef'"
    >
        <el-affix
            v-if="!db.isRepresented"
            :target="`#single-dashboard-${props.singleDbId}`"
            :offset="affixOffset"
        >
            <div class="single-dashboard-head">
                <div class="title">Dashboard ({{ db.name || "" }})</div>
                <span>
                    <el-button
                        type="primary"
                        :disabled="db ? isEmptyDict(db.nodesSelections) : true"
                        @click="handleFilter"
                    >
                        <filterSvg
                            style="
                                height: 1.5em;
                                width: 1.5em;
                                margin-right: 0.5em;
                            "
                        />
                        FILTER!
                    </el-button>
                    <el-divider
                        v-show="db ? db.clearSelMode === 'manual' : false"
                        direction="vertical"
                    ></el-divider>
                    <el-popover
                        placement="bottom-end"
                        trigger="hover"
                        :width="400"
                    >
                        <template #reference>
                            <el-button
                                v-show="
                                    db ? db.clearSelMode === 'manual' : false
                                "
                                @click="handleClearAll"
                            >
                                <clearSvg
                                    style="
                                        height: 1.5em;
                                        width: 1.5em;
                                        margin-right: 0.5em;
                                    "
                                />clear selection
                            </el-button>
                        </template>
                        <div style="text-align: start">
                            <h3>Which selections do you want to clear?</h3>

                            <div
                                v-for="d in Object.keys(db.nodesSelections)"
                                :key="d"
                            >
                                <el-tooltip effect="light" placement="top">
                                    <template #content>
                                        <div
                                            :style="{
                                                maxWidth: '300px',
                                                color: myStore
                                                    .defaultNodesSelectionEntryDescriptions[
                                                    d
                                                ]
                                                    ? 'currentcolor'
                                                    : 'red',
                                            }"
                                        >
                                            {{
                                                myStore
                                                    .defaultNodesSelectionEntryDescriptions[
                                                    d
                                                ] || "description undefined"
                                            }}
                                        </div>
                                    </template>
                                    <div class="clear-item">
                                        <span>{{ d }}</span>
                                        <el-button
                                            :disabled="
                                                d === 'full' ||
                                                isEmptyDict(
                                                    db.nodesSelections[d]
                                                )
                                            "
                                            @click="(e:Event) => handleClearOne(e, d)"
                                            >clear
                                        </el-button>
                                    </div>
                                </el-tooltip>
                            </div>
                        </div>
                    </el-popover>
                </span>
                <span>
                    <el-button @click="handleRestoreViewsSizes">
                        <restoreViewsSizesSvg
                            style="
                                height: 1.5em;
                                width: 1.5em;
                                margin-right: 0.5em;
                            "
                        />
                        restore views sizes
                    </el-button>
                    <el-divider direction="vertical" />

                    <el-popover
                        placement="bottom-end"
                        trigger="hover"
                        :width="500"
                        @after-enter="
                            async () => {
                                await myStore.calcSnapshotsOfDashboard(db);
                            }
                        "
                        :disabled="myStore.recentSingleDashboardList.length < 2"
                    >
                        <template #reference>
                            <el-button
                                :disabled="
                                    myStore.recentSingleDashboardList.length < 2
                                "
                                @click="handleRepresent"
                            >
                                <representSvg
                                    style="
                                        height: 1.5em;
                                        width: 1.5em;
                                        margin-right: 0.5em;
                                    "
                                />
                                represent
                            </el-button>
                        </template>
                        <div style="text-align: start">
                            <h3>Views Snapshots</h3>
                            <div>
                                (Click button to auto select a view by algo,
                                click below snapshots to manually select a view)
                            </div>
                            <el-divider />

                            <el-scrollbar :max-height="500">
                                <div
                                    :style="{
                                        margin: '0 auto',
                                        display: 'grid',
                                        gap: '20px',
                                        gridTemplateColumns: '202px 202px',

                                        gridTemplateRows: `repeat(${Math.ceil(
                                            db.viewsDefinitionList.length / 2
                                        )}, 1fr)`,
                                        justifyContent: 'center',
                                    }"
                                >
                                    <div
                                        v-for="view in db.viewsDefinitionList"
                                        :key="view.viewName"
                                        :style="{
                                            border: '1px solid black',
                                            boxSizing: 'content-box',
                                            height:
                                                (view.bodyHeight * 200) /
                                                    view.bodyWidth +
                                                2 +
                                                'px',
                                            cursor: 'pointer',
                                        }"
                                        @click="
                                            () => {
                                                db.fromViewName = view.viewName;
                                                handleRepresent(undefined);
                                            }
                                        "
                                    >
                                        <LoadingComp
                                            v-if="view.isGettingSnapshot"
                                            :text="`loading snapshot of view ${view.viewName}`"
                                        />
                                        <ErrorComp
                                            v-else-if="
                                                view.gettingSnapShotError
                                            "
                                            :error="view.gettingSnapShotError"
                                        />
                                        <img
                                            v-else-if="view.snapshotBase64"
                                            :style="{
                                                width: '200px',
                                            }"
                                            :src="view.snapshotBase64"
                                            :alt="`the snapshot of view ${view.viewName}`"
                                        />
                                    </div>
                                </div>
                            </el-scrollbar>
                        </div>
                    </el-popover>
                </span>
                <span>
                    <el-popover
                        placement="bottom-end"
                        trigger="hover"
                        :width="400"
                    >
                        <template #reference>
                            <legendSvg ref="legendSvgRef" />
                        </template>
                        <ScatterSymbolLegend />
                    </el-popover>
                    <el-divider direction="vertical" />
                    <el-popover
                        placement="bottom-end"
                        :trigger="myStore.settingsMenuTriggerMode"
                        :width="400"
                    >
                        <template #reference>
                            <settingSvg
                                style="
                                    height: 2em;
                                    width: 2em;
                                    color: var(--el-text-color-regular);
                                "
                            />
                        </template>
                        <div style="text-align: start">
                            <h3>Dashboard Settings</h3>
                            <el-divider />

                            <span class="setting-item-name"
                                >Label type:(only this dashboard){{ " " }}
                            </span>
                            <el-tooltip
                                effect="dark"
                                placement="top"
                                :z-index="9999"
                            >
                                <template #content>
                                    <div
                                        v-if="
                                            ds.taskType ===
                                            'node-classification'
                                        "
                                    >
                                        true or pred label for each node
                                    </div>
                                    <div
                                        v-else-if="
                                            ds.taskType === 'link-prediction'
                                        "
                                    >
                                        disabled, since all node labels are true
                                        labels
                                    </div>
                                    <div
                                        v-else-if="
                                            ds.taskType ===
                                            'graph-classification'
                                        "
                                    >
                                        true or pred label for each graph,<br />
                                        all node labels are true labels
                                    </div>
                                </template>
                                <el-switch
                                    :disabled="
                                        ds.taskType === 'link-prediction'
                                    "
                                    v-model="db.labelType"
                                    active-text="pred label"
                                    inactive-text="true label"
                                    :active-value="'pred'"
                                    :inactive-value="'true'"
                                />
                            </el-tooltip>
                            <br />

                            <span class="setting-item-name">
                                Select mode:{{ " " }}
                            </span>
                            <el-switch
                                v-model="db.clearSelMode"
                                :active-value="clearSelModes[0]"
                                :inactive-value="clearSelModes[1]"
                                active-text="manual clear (merge select)"
                                inactive-text="auto clear (re-select)"
                            >
                            </el-switch>
                            <br />

                            <span class="setting-item-name"
                                >Rescale coords when:{{ " " }}
                            </span>
                            <el-switch
                                v-model="db.whenToRescaleMode"
                                :active-value="whenToRescaleModes[1]"
                                :inactive-value="whenToRescaleModes[0]"
                                active-text="on resize end"
                                inactive-text="simultaneously"
                            >
                            </el-switch>
                            <br />

                            <span class="setting-item-name"
                                >Highlight corresponding node:{{ " " }}</span
                            >
                            <el-switch
                                v-model="db.isHighlightCorrespondingNode"
                            />
                            <br />
                        </div>
                    </el-popover>
                </span>
            </div>
        </el-affix>

        <div v-else class="represented-single-dashboard-head">
            <el-tooltip
                :visible="isTooltipVisible"
                effect="dark"
                placement="top"
                :z-index="9999"
            >
                <template #content> 👋 Drag here!</template>
                <span
                    ref="handle"
                    class="cursor-move"
                    @mouseenter="isTooltipVisible = true"
                    @mouseleave="isTooltipVisible = false"
                >
                    <handleSvg />
                    {{ " " }}
                    {{ db.name }}
                </span>
            </el-tooltip>
            <span>
                <el-button @click="handleMaximize">
                    <maximizeSvg style="height: 1.5em; width: 1.5em" />
                </el-button>
                <el-button @click="handleMinimize">
                    <minimizeSvg style="height: 1.5em; width: 1.5em" />
                </el-button>
            </span>
        </div>
        <div
            :class="
                db.isRepresented
                    ? 'represented-single-dashboard-body'
                    : 'single-dashboard-body'
            "
        >
            <!-- <LoadingComp v-if="isLoading" :text="loadingText" />
            <ErrorComp v-else-if="loadingError" :error="loadingError" /> -->
            <draggable
                v-model="db.viewsDefinitionList"
                v-bind="dragOptions"
                handle=".handle"
                item-key="viewName"
                :class="
                    db.isRepresented
                        ? 'represented-draggable-container'
                        : 'draggable-container'
                "
                @end="isDragging = false"
                @start="isDragging = true"
                ref="draggableContainerRef"
            >
                <!--NOTE .handle可能在孙组件-->
                <!--NOTE element就是viewsDefinitionList的每个元素，childCompObj则是别名-->
                <template
                    #item="{
                        element: childCompObj,
                        // index //暂时用不到
                    }"
                >
                    <ResizableBox
                        class="list-group-item"
                        :dbId="props.singleDbId"
                        :viewName="childCompObj.viewName"
                        v-show="
                            !db.isRepresented ||
                            childCompObj.viewName === db.fromViewName
                        "
                        :ref="
                            childCompObj.viewName === db.fromViewName
                                ? 'resizableBoxRef'
                                : 'otherResizableBoxRef'
                        "
                    >
                        <template #header>
                            <PublicHeader
                                :dbId="props.singleDbId"
                                :viewName="childCompObj.viewName"
                            />
                            <component
                                :is="childCompObj.headerComp"
                                :dbId="props.singleDbId"
                                :viewName="childCompObj.viewName"
                                v-bind="childCompObj.headerProps"
                            />
                        </template>

                        <template #default>
                            <!-- <KeepAlive> -->
                            <component
                                :is="childCompObj.bodyComp"
                                :dbId="props.singleDbId"
                                :viewName="childCompObj.viewName"
                                v-bind="childCompObj.bodyProps"
                            />
                            <!-- </KeepAlive> -->
                        </template>
                    </ResizableBox>
                </template>
            </draggable>
        </div>
    </div>
</template>

<script setup lang="ts">
import {
    ref,
    watch,
    shallowRef,
    toRaw,
    computed,
    defineAsyncComponent,
    onActivated,
    onDeactivated,
    onUnmounted,
    onMounted,
} from "vue";

import type {
    DenseView,
    Node,
    Link,
    NodeCoord,
    SparseView,
    Type_LinkId,
    Dashboard,
    Type_NodeId,
    Type_GraphId,
    Type_NodesSelectionEntryId,
} from "@/types/types";

import LoadingComp from "./state/Loading.vue";
import ErrorComp from "./state/Error.vue";

import { Edit } from "@element-plus/icons-vue";
import { default as settingSvg } from "@/components/icon/FluentSettings48Regular.vue";
import { default as restoreViewsSizesSvg } from "./icon/MdiMoveResize.vue";
import { default as clearSvg } from "./icon/MdiSelectionRemove.vue";
import { default as filterSvg } from "./icon/FluentSquareHintSparkles32Filled.vue";
import { default as representSvg } from "./icon/MdiMonitorStar.vue";
import { default as handleSvg } from "@/components/icon/RadixIconsDragHandleHorizontal.vue";
import { default as minimizeSvg } from "@/components/icon/PrimeWindowMinimize.vue";
import { default as maximizeSvg } from "@/components/icon/SolarMaximizeSquare2Outline.vue";
import { default as legendSvg } from "@/components/icon/GisMapLegend.vue";

import draggable from "vuedraggable";
import ResizableBox from "./publicViews/resizableBox.vue";
import PublicHeader from "./publicViews/publicHeader.vue";
import { isClient } from "@vueuse/shared";
import { useDraggable } from "@vueuse/core";
import { useElementBounding } from "@vueuse/core";

import { useMyStore } from "@/stores/store";
import { clearSelModes, whenToRescaleModes } from "@/stores/enums";
import { useWebWorkerFn } from "@/utils/myWebWorker";
import BitSet from "bitset";
import {
    isEmptyDict,
    calcGraphCoords,
    nodeMapGraph2GraphMapNodes,
    calcNeighborDict,
} from "@/utils/graphUtils";
import { customAlphabet } from "nanoid";

import GraphRenderer from "./publicViews/graphRenderer.vue";
import TsneHeader from "./publicViews/tsneHeader.vue";
import TsneRenderer from "./publicViews/tsneRenderer.vue";
import TopoLatentDensityRenderer from "./publicViews/topoLatentDensityRenderer.vue";
import TopoLatentDensityHeader from "./publicViews/topoLatentDensityHeader.vue";
import ConfusionMatrixRenderer from "./publicViews/confusionMatrixRenderer.vue";
import SparseFeatureRenderer from "./publicViews/sparseFeatureRenderer.vue";
import DenseFeatureRenderer from "./publicViews/denseFeatureRenderer.vue";
import DenseFeatureHeader from "./publicViews/denseFeatureHeader.vue";
import SparseFeatureHeader from "./publicViews/sparseFeatureHeader.vue";
import LinkPredRenderer from "./publicViews/linkPredRenderer.vue";
import LinkPredHeader from "./publicViews/linkPredHeader.vue";
import MultiGraphRenderer from "./publicViews/multiGraphRenderer.vue";
import MultiGraphHeader from "./publicViews/multiGraphHeader.vue";
import GraphFeatureRenderer from "./publicViews/graphFeatureRenderer.vue";
import GraphFeatureHeader from "./publicViews/graphFeatureHeader.vue";
import GraphTsneRenderer from "./publicViews/graphTsneRenderer.vue";
import GraphTsneHeader from "./publicViews/graphTsneHeader.vue";
import GraphHeader from "./publicViews/graphHeader.vue";
import ScatterSymbolLegend from "./header/scatterSymbolLegend.vue";

const props = defineProps({
    singleDbId: {
        type: String,
        required: true,
        default: "",
    },
});
const myStore = useMyStore();

const isLoading = ref(true);
const loadingText = ref<string>("");
const loadingError = ref<Error>();

const db = myStore.getSingleDashboardById(props.singleDbId)!;
console.log("in single db, props id is", props.singleDbId);
console.log("in single db, db is", db);
const ds = myStore.getDatasetByName(db.refDatasetName)!;

onMounted(() => {
    console.warn("in single Db, onMounted!");
});
onUnmounted(() => {
    console.warn("in single Db, onUnmounted!");
});

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION data on dashboard created
const calcSubCoords = async () => {
    /////////////////////////////////////////
    ////// calc force-directed layout
    const {
        workerFn: graphWorkerFn,
        // workerStatus: graphWorkerStatus,
        workerTerminate: graphWorkerTerminate,
    } = useWebWorkerFn(
        calcGraphCoords<
            Node,
            Link & d3.SimulationLinkDatum<Node & d3.SimulationNodeDatum>
        >,
        {
            timeout: 20_000,
            dependencies: [
                // "http://localhost:5173/workers/d3js.org_d3.v7.js",
                // "https://d3js.org/d3.v7.js", //for debug
                "https://d3js.org/d3.v7.min.js",
            ],
        }
    );
    const localNodes =
        toRaw(ds.nodes)?.filter((d) => db.srcNodesDict[d.id]) || [];
    const localLinks =
        toRaw(ds.links)?.filter(
            (d) => db.srcNodesDict[d.source] && db.srcNodesDict[d.target]
        ) || [];
    console.log(
        "in singDb, in calcSubCoords, local Nodes&Links",
        localNodes,
        localLinks
    );
    db.srcLinksDict = localLinks.reduce(
        (acc, cur) => ({
            ...acc,
            [cur.eid]: cur.gid,
        }),
        {}
    );
    db.srcLinksArr = localLinks;
    db.graphCoordsRet = await graphWorkerFn(localNodes, localLinks);
    console.log(
        "in singleDb",
        props.singleDbId,
        "in calcSubCoords, useWebWorker, got graphCoordsRet: ",
        db.graphCoordsRet
    );
    graphWorkerTerminate();
    ////// calc force-directed layout
    /////////////////////////////////////////

    const rootDb = myStore.singleDashboardList.find((d) => d.isRoot);
    db.tsneRet = rootDb?.tsneRet.filter((d) => db.srcNodesDict[d.id]) || [];
    if (rootDb?.graphTsneRet) {
        const graphMapNodes = nodeMapGraph2GraphMapNodes(db.srcNodesDict);
        db.graphTsneRet = rootDb.graphTsneRet.filter(
            (d) => graphMapNodes[d.id]
        );
    }
};
const loadDbData = async () => {
    console.log("load SingleDb Data!");
    if (db) {
        db.nodesSelections["full"] = JSON.parse(
            JSON.stringify(db.srcNodesDict)
        ); //{...undefined} = {}
        if (db.isRoot && db.isComplete) {
            // REVIEW nothing need?
            // console.log("in loadDbData, db.srcNodesDict", db.srcNodesDict);
            // console.log( "in loadDbData, after assign db.nodesSelections", db.nodesSelections, "db.nodesSelections['full']", db.nodesSelections["full"]);
            return;
        } else {
            await calcSubCoords();
        }
    }
};

//on Created
(() => {
    if (db) {
        db.calcOnCreatedPromise = new Promise<void>((resolve, reject) => {
            // isLoading.value = true;
            // loadingError.value = undefined;
            // loadingText.value = "loading dashboard data";

            //某些个缺少某个view的数据集。例如没有nodeFeat的，可以在loadDataset的时候就修改defaultVwNames
            //这个也是一种策略，
            db.viewsDefinitionList.forEach((view) => {
                view.setAttr("bodyComp", shallowRef(LoadingComp)).setAttr(
                    "bodyProps",
                    {
                        text: db.isRoot ? "loading data" : "loading sub data",
                    }
                );
            });

            loadDbData()
                .then(() => {
                    console.log("in single Db, loadDbData() finished!");
                    const viewLatent = myStore
                        .getViewByName(db, "Latent Space")
                        ?.setAttr("headerComp", shallowRef(TsneHeader))
                        .setAttr("bodyComp", shallowRef(TsneRenderer))
                        .setAttr("headerProps", { which: 0 })
                        .setAttr("bodyProps", { which: 0 });
                    const viewTopoLatentDensity = myStore
                        .getViewByName(db, "Topo + Latent Density")
                        ?.setAttr(
                            "bodyComp",
                            shallowRef(TopoLatentDensityRenderer)
                        )
                        .setAttr("bodyProps", { which: 0 })
                        .setAttr(
                            "headerComp",
                            shallowRef(TopoLatentDensityHeader)
                        )
                        .setAttr("headerProps", { which: 0 });
                    if (ds.taskType === "node-classification") {
                        const viewTopo = myStore
                            .getViewByName(db, "Topology Space")
                            ?.setAttr("bodyComp", shallowRef(GraphRenderer))
                            .setAttr("headerComp", shallowRef(GraphHeader));
                        const viewPred = myStore
                            .getViewByName(db, "Prediction Space")
                            ?.setAttr(
                                "bodyComp",
                                shallowRef(ConfusionMatrixRenderer)
                            )
                            .setAttr("bodyProps", { which: 0 });
                    } else if (ds.taskType === "link-prediction") {
                        const viewTopo = myStore
                            .getViewByName(db, "Topology Space")
                            ?.setAttr("bodyComp", shallowRef(GraphRenderer))
                            .setAttr("headerComp", shallowRef(GraphHeader));
                        const viewPred = myStore
                            .getViewByName(db, "Prediction Space")
                            ?.setAttr("bodyComp", shallowRef(LinkPredRenderer))
                            .setAttr("bodyProps", { which: 0 })
                            .setAttr("headerComp", shallowRef(LinkPredHeader))
                            .setAttr("headerProps", { which: 0 });
                    } else if (ds.taskType === "graph-classification") {
                        //  graph topo
                        const viewTopo = myStore
                            .getViewByName(db, "Topology Space")
                            ?.setAttr("initialWidth", 1000)
                            .setAttr("initialHeight", 600)
                            .setAttr("bodyComp", shallowRef(MultiGraphRenderer))
                            .setAttr("headerComp", shallowRef(MultiGraphHeader))
                            .setAttr("bodyProps", {});

                        //  graph feature
                        if (db.isRoot) {
                            const viewGraphFeat = myStore.initialNewView(
                                "Feature Space - Graph"
                            ) as DenseView;
                            myStore.defaultNodesSelectionEntryMapper[
                                "Feature Space - Graph"
                            ] = {
                                source: ["full"],
                                target: ["public"],
                            };
                            viewGraphFeat
                                .setAttr(
                                    "bodyComp",
                                    shallowRef(GraphFeatureRenderer)
                                )
                                .setAttr(
                                    "headerComp",
                                    shallowRef(GraphFeatureHeader)
                                )
                                .setAttr("initialWidth", 250)
                                .setAttr("initialHeight", 500);
                            myStore.insertViewAfterName(
                                db,
                                "Topology Space",
                                viewGraphFeat
                            );
                        } else {
                            myStore
                                .getViewByName(db, "Feature Space - Graph")
                                ?.setAttr(
                                    "bodyComp",
                                    shallowRef(GraphFeatureRenderer)
                                )
                                .setAttr(
                                    "headerComp",
                                    shallowRef(GraphFeatureHeader)
                                )
                                .setAttr("initialWidth", 600);
                        }
                        // graph pred
                        const viewPred = myStore
                            .getViewByName(db, "Prediction Space")
                            ?.setAttr(
                                "bodyComp",
                                shallowRef(ConfusionMatrixRenderer)
                            )
                            .setAttr("bodyProps", { which: 0 });

                        myStore
                            .getViewByName(db, "Graph Latent Space")
                            ?.setAttr("headerComp", shallowRef(GraphTsneHeader))
                            .setAttr("bodyComp", shallowRef(GraphTsneRenderer))
                            .setAttr("headerProps", { which: 0 })
                            .setAttr("bodyProps", { which: 0 });
                    }

                    if (ds.nodeSparseFeatureValues) {
                        //REVIEW how to judge a dense feat or sparse feat

                        if (db.isRoot) {
                            const sparseView = myStore.initialNewView(
                                "Feature Space - Sparse"
                            ) as SparseView;
                            myStore.defaultNodesSelectionEntryMapper[
                                "Feature Space - Sparse"
                            ] = {
                                source: ["full"],
                                target: ["public"],
                            };
                            sparseView
                                .setAttr(
                                    "bodyComp",
                                    shallowRef(SparseFeatureRenderer)
                                )
                                .setAttr(
                                    "headerComp",
                                    shallowRef(SparseFeatureHeader)
                                )
                                .setAttr("bodyMargins", {
                                    top: 0.03,
                                    left: 0.03,
                                    right: 0.03,
                                    bottom: 0.03,
                                });
                            myStore.insertViewAfterName(
                                db,
                                db.viewsDefinitionList.at(-1)!.viewName,
                                sparseView
                            );
                        } else {
                            myStore
                                .getViewByName(db, "Feature Space - Sparse")
                                ?.setAttr(
                                    "bodyComp",
                                    shallowRef(SparseFeatureRenderer)
                                )
                                .setAttr(
                                    "headerComp",
                                    shallowRef(SparseFeatureHeader)
                                )
                                .setAttr("bodyMargins", {
                                    top: 0.03,
                                    left: 0.03,
                                    right: 0.03,
                                    bottom: 0.03,
                                });
                        }
                    }
                    if (ds.denseNodeFeatures) {
                        if (db.isRoot) {
                            const denseView = myStore.initialNewView(
                                "Feature Space - Dense"
                            ) as DenseView;
                            myStore.defaultNodesSelectionEntryMapper[
                                "Feature Space - Dense"
                            ] = {
                                source: ["full"],
                                target: ["public"],
                            };
                            // NOTE 若给mapper添加了db.nodesSelection没有的entry名称，需要db.nodesSelection[newEntry] = {}
                            denseView
                                .setAttr(
                                    "bodyComp",
                                    shallowRef(DenseFeatureRenderer)
                                )
                                .setAttr(
                                    "headerComp",
                                    shallowRef(DenseFeatureHeader)
                                )
                                .setAttr("initialWidth", 800);
                            myStore.insertViewAfterName(
                                db,
                                db.viewsDefinitionList.at(-1)!.viewName,
                                denseView
                            );
                        } else {
                            myStore
                                .getViewByName(db, "Feature Space - Dense")
                                ?.setAttr(
                                    "bodyComp",
                                    shallowRef(DenseFeatureRenderer)
                                )
                                .setAttr(
                                    "headerComp",
                                    shallowRef(DenseFeatureHeader)
                                )
                                .setAttr("initialWidth", 800);
                        }
                    }

                    // console.log(
                    //     "in singleDB, in loadDbData finished ,then(), get views' initialWidth:",
                    //     viewTopo?.initialWidth,
                    //     viewTopo?.initialHeight,
                    //     viewLatent?.initialWidth,
                    //     viewLatent?.initialHeight
                    // );
                    resolve();

                    // isLoading.value = false;
                })
                .catch((err) => {
                    reject(err);

                    // isLoading.value = false;
                    // loadingError.value = err;
                });
        });
    }
})(); //onCreated

////////////////// !SECTION data on dashboard created
////////////////////////////////////////////////////////////////////////////////

const handleFilter = async (e: Event) => {
    const selAndNeighbor = calcNeighborDict(
        db.nodesSelections["public"],
        ds.hops!,
        ds.neighborMasksByHop!
    );
    const ret: Record<
        Type_NodeId,
        { gid: Type_GraphId; parentDbIndex: number; hop: number }
    > = {};
    for (const id in selAndNeighbor) {
        ret[id] = {
            ...selAndNeighbor[id],
            parentDbIndex: 0,
            gid: ds.globalNodesDict[id].gid,
        };
    }

    await myStore.calcPrincipalSnapshotOfDashboard(db);

    const nanoid = customAlphabet("abcdefghijklmnopqrstuvwxyz_", 10);
    myStore.addSingleDashboard(
        {
            id: nanoid(), //=> "不含数字"
            refDatasetName: db.refDatasetName,
            date: Date.now ? Date.now() : new Date().getTime(),
            isRoot: false,
            isComplete: false, //REVIEW
            parentId: db.id,
            fromViewName: myStore.defaultSingleViewNames[ds.taskType][0],
            graphCoordsRet: undefined,
            srcNodesDict: {
                // ...db.nodesSelections["public"],
                ...ret,
            },
        },
        db.viewsDefinitionList.map((d) => d.viewName)
    );
    myStore.repairButtonFocus(e);
};

const handleClearAll = (e: Event) => {
    for (const key in db.nodesSelections) {
        if (key !== "full") {
            db.nodesSelections[key] = {};
        }
    }
    db.viewsDefinitionList.forEach((view) => {
        view.hideRectWhenClearSelFunc();
    });
    myStore.repairButtonFocus(e);
};
const handleClearOne = (e: Event, key: Type_NodesSelectionEntryId) => {
    db.nodesSelections[key] = {};
    db.viewsDefinitionList.forEach((view) => {
        if (
            myStore
                .getViewTargetNodesSelectionEntry(view.viewName)
                .includes(key)
        )
            view.hideRectWhenClearSelFunc();
    });
    myStore.repairButtonFocus(e);
};

const handleRestoreViewsSizes = (e: Event) => {
    db.restoreViewsSizesSignal = !db.restoreViewsSizesSignal;
    myStore.repairButtonFocus(e);
};

// represent related
const handleRepresent = async (e: Event | undefined) => {
    //assume len >=2
    // const { id } = myStore.recentSingleDashboardList.pop()!;
    // myStore.representedSingleDashboardList.push(id);

    await myStore.calcPrincipalSnapshotOfDashboard(props.singleDbId);
    db.isRepresented = true;
    const { id } = myStore.recentSingleDashboardList.at(-2)!;
    myStore.toSingleDashboardById(id);
    // myStore.recentSingleDashboardList.at(-1)!.isRepresented = true;
    if (e) myStore.repairButtonFocus(e);
};
const representedDashboardRef = ref<HTMLDivElement | null>(null);
const dashboardRef = ref<HTMLDivElement | null>(null);

const isTooltipVisible = ref(true);

const topOffset = computed(() => window?.innerHeight * 0.05 || 50);
const bottomOffset = computed(() => window?.innerHeight * 0.95 || 50);
const initialWidth = 400;
const initialHeight = 470;
const handle = ref<HTMLElement | null>(null);
const innerWidth = isClient ? window.innerWidth : 200;
const isOnMove = ref(false);
const { x, y, style } = useDraggable(representedDashboardRef, {
    preventDefault: true,
    handle: handle,
    initialValue: { x: innerWidth / 4.2, y: 80 },
    onStart: () => {
        isOnMove.value = true;
        if (db.isRepresented) {
            isTooltipVisible.value = false;
            return;
        } else {
            return false;
        }
    },
    onMove: () => {
        isOnMove.value = true;
    },
    onEnd: (position, pointerEvent) => {
        // isTooltipVisible.value = true;

        isOnMove.value = false;
        if (position.x <= 0) x.value = 0;
        if (position.y <= topOffset.value) y.value = topOffset.value;
        else if (position.y + dragDbHeight.value >= bottomOffset.value)
            y.value = bottomOffset.value - dragDbHeight.value;
    },
});
const {
    y: dragDbY,
    x: dragDbX,
    top: dragDbTop,
    right: dragDbRight,
    bottom: dragDbBottom,
    left: dragDbLeft,
    width: dragDbWidth,
    height: dragDbHeight,
} = useElementBounding(representedDashboardRef);

const computedWidthStr = computed(() => initialWidth + "px");
const computedHeightStr = computed(() => initialHeight + "px");
const borderWidth = 1;
const borderWidthStr = computed(() => borderWidth + "px");
const principalView = myStore.getPrincipalViewOfDashboard(db);

const resizableBoxRef = ref<InstanceType<typeof ResizableBox> | null>(null);

watch(
    [dragDbWidth, dragDbHeight],
    ([newW, newH]) => {
        if (resizableBoxRef.value && db.isRepresented && !isOnMove.value) {
            resizableBoxRef.value.heightStr =
                (newH - 2 * borderWidth) * 0.88 + "px";
            resizableBoxRef.value.widthStr = newW * 0.95 + "px";
        }
    },
    { immediate: true }
);
const handleMaximize = (e: Event) => {
    // myStore.toSingleDashboardById(props.singleDbId);
    db.isRepresented = false;
    myStore.toSingleDashboardById(props.singleDbId);
    myStore.repairButtonFocus(e);
};
const handleMinimize = (e: Event) => {
    db.isRepresented = false;
    const i = myStore.recentSingleDashboardList.findIndex(
        (d) => d.id === props.singleDbId
    );

    if (i < 0) {
        throw new Error(`in representedDb failed to find ${props.singleDbId}!`);
    } else {
        if (i < myStore.recentSingleDashboardList.length - myStore.recentLen) {
            //说明renderList里面没有，应当更新renderIndex
            myStore.recentSingleDashboardList[i].renderIndex =
                myStore.recentSingleDashboardList.at(
                    -myStore.recentLen
                )!.renderIndex;
            myStore.recentSingleDashboardList.at(
                -myStore.recentLen
            )!.renderIndex = 0;
        }
        const to = myStore.recentSingleDashboardList.splice(i, 1); //这里是个数组;
        // if (to[0].isRepresented) to[0].isRepresented = false;

        myStore.recentSingleDashboardList.splice(-1, 0, to[0]); //插到倒数第二个，代表优先级次优
    }
    myStore.repairButtonFocus(e);
};
// draggable相关
const dragOptions = ref({
    animation: 500,
    // group: "description",
    disabled: false,
    ghostClass: "ghost",
});
const isDragging = ref(false);

//affix悬浮效果相关
const affixOffset = computed(() => window?.innerHeight * 0.05 || 50);
</script>

<style scoped>
.clear-item {
    padding-inline-start: 1em;
    display: flex;
    flex-direction: row;
    justify-content: space-between;
}

.clear-item:hover {
    background-color: LightGray;
}
.ghost {
    /* for draggable */
    opacity: 0.5;
    background: #c8ebfb;
}

.represented-single-dashboard-head {
    margin: 0;
    background-color: white;
    padding: 0.25em 20px;
    /* width: 100%; */
    height: 8%;
    border-bottom: 1px solid rgba(0, 0, 0, 0.16);
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
}

.represented-single-dashboard-body {
    margin: 0;
    padding: 0;
    opacity: 1;
    background-color: white;
    height: 92%;
}

.represented-single-dashboard {
    position: fixed;

    z-index: 8800;

    resize: both;
    overflow: hidden;
    /* width: v-bind("viewDef.initialWidth + 'px'"); */
    width: v-bind(computedWidthStr);
    height: v-bind(computedHeightStr);

    /* box-sizing: border-box; */
    box-sizing: content-box;

    border-radius: 2px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.12), 0 0 6px rgba(0, 0, 0, 0.04);
    border: solid rgba(0, 0, 0, 0.12);
    border-width: v-bind(borderWidthStr);

    /* user-select: none; */
}

.single-dashboard {
    padding: 0;
    margin: 0;
    /* position: relative; */ /* REVIEW */
    border: 1px solid rgba(0, 0, 0, 0.16);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.16), 0 0 6px rgba(0, 0, 0, 0.08);
}

.single-dashboard-head {
    background-color: white;
    padding: 0.25em 20px;
    width: calc(100% - 40px);
    min-height: 3em;
    border-bottom: 1px solid rgba(0, 0, 0, 0.16);

    display: flex;
    justify-content: space-between;
    align-items: center;
}
.title {
    font-weight: bold;
}
.setting-item-name {
    font-weight: bold;
}

.represented-draggable-container {
    margin: 0;
    padding: 0;

    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    justify-content: flex-start;
    align-items: start;
}
.represented-draggable-container > * {
    margin: auto;
}
.draggable-container {
    margin: 0;
    padding-bottom: 20px;
    padding-right: 20px;

    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    justify-content: flex-start;
    align-items: start;
}

.draggable-container > * {
    margin-top: 20px;
    margin-left: 20px;
}
.cursor-move {
    cursor: move;
}
</style>
