<template>
    <div
        :id="props.compDbId"
        :style="style"
        :class="{
            'comp-dashboard': !db.isRepresented,
            'represented-comp-dashboard': db.isRepresented,
        }"
        :ref="db.isRepresented ? 'representedDashboardRef' : 'dashboardRef'"
    >
        <el-affix
            v-if="!db.isRepresented"
            target=".comp-dashboard"
            :offset="affixOffset"
        >
            <div class="comp-dashboard-head">
                <div class="title">
                    Comparative Dashboard ({{ db.name || "" }})
                </div>

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
                                            @click="(e) => handleClearOne(e, d)"
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
                        :disabled="myStore.recentCompDashboardList.length < 2"
                    >
                        <template #reference>
                            <el-button
                                :disabled="
                                    myStore.recentCompDashboardList.length < 2
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
                            <setting
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
                                >Label type(only this dashboard):{{ " " }}
                            </span>
                            <el-tooltip
                                effect="dark"
                                placement="top"
                                :z-index="9999"
                            >
                                <template #content>
                                    <div
                                        v-if="
                                            ds1.taskType ===
                                            'node-classification'
                                        "
                                    >
                                        true or pred label for each node
                                    </div>
                                    <div
                                        v-else-if="
                                            ds1.taskType === 'link-prediction'
                                        "
                                    >
                                        disabled, since all node labels are true
                                        labels
                                    </div>
                                    <div
                                        v-else-if="
                                            ds1.taskType ===
                                            'graph-classification'
                                        "
                                    >
                                        true or pred label for each graph,<br />
                                        all node labels are true labels
                                    </div>
                                </template>
                                <el-switch
                                    :disabled="
                                        ds1.taskType === 'link-prediction'
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
        <div v-else class="represented-comp-dashboard-head">
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

        <!-- <div v-if="isLoading" class="comp-dashboard-body">Loading</div> -->
        <div
            :class="
                db.isRepresented
                    ? 'represented-comp-dashboard-body'
                    : 'comp-dashboard-body'
            "
        >
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
                        :dbId="props.compDbId"
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
                                :dbId="props.compDbId"
                                :viewName="childCompObj.viewName"
                            />
                            <component
                                :is="childCompObj.headerComp"
                                :dbId="props.compDbId"
                                :viewName="childCompObj.viewName"
                                v-bind="childCompObj.headerProps"
                            />
                        </template>

                        <template #default>
                            <component
                                :is="childCompObj.bodyComp"
                                :dbId="props.compDbId"
                                :viewName="childCompObj.viewName"
                                v-bind="childCompObj.bodyProps"
                            />
                        </template>
                    </ResizableBox>
                </template>
            </draggable>
        </div>
    </div>
</template>

<script setup lang="ts">
import {
    h,
    ref,
    watch,
    shallowRef,
    toRaw,
    defineAsyncComponent,
    onMounted,
    computed,
    onUnmounted,
    onActivated,
    onDeactivated,
} from "vue";
import { useMyStore } from "@/stores/store";
import { clearSelModes, whenToRescaleModes } from "@/stores/enums";
import type {
    Type_RankEmbDiffAlgos,
    Type_polarEmbDiffAlgos,
    Type_polarTopoDistAlgos,
    CompDashboard,
    View,
    RankView,
    PolarView,
    DenseView,
    SparseView,
    Type_NodeId,
    Type_GraphId,
    Type_NodesSelectionEntryId,
} from "@/types/types";

import { default as setting } from "@/components/icon/FluentSettings48Regular.vue";
import { ElButton } from "element-plus";
import { default as restoreViewsSizesSvg } from "./icon/MdiMoveResize.vue";
import { default as representSvg } from "./icon/MdiMonitorStar.vue";
import { default as handleSvg } from "@/components/icon/RadixIconsDragHandleHorizontal.vue";
import { default as minimizeSvg } from "@/components/icon/PrimeWindowMinimize.vue";
import { default as maximizeSvg } from "@/components/icon/SolarMaximizeSquare2Outline.vue";
import { default as clearSvg } from "./icon/MdiSelectionRemove.vue";
import { default as filterSvg } from "./icon/FluentSquareHintSparkles32Filled.vue";
import { default as legendSvg } from "@/components/icon/GisMapLegend.vue";

import draggable from "vuedraggable";
import { nanoid } from "nanoid";
import BitSet from "bitset";
import { isClient } from "@vueuse/shared";
import { useDraggable } from "@vueuse/core";
import { useElementBounding } from "@vueuse/core";

import {
    isEmptyDict,
    calcRank3,
    calcPolar,
    calcGraphCoords,
    nodeMapGraph2GraphMapNodes,
    calcNeighborDict,
} from "../utils/graphUtils";
import { useWebWorkerFn } from "@/utils/myWebWorker";

import LoadingComp from "./state/Loading.vue";
import PendingComp from "./state/Pending.vue";
import ErrorComp from "./state/Error.vue";
import ResizableBox from "./publicViews/resizableBox.vue";
import PublicHeader from "./publicViews/publicHeader.vue";

import TsneHeader from "./publicViews/tsneHeader.vue";
import TsneRenderer from "./publicViews/tsneRenderer.vue";
import GraphHeader from "./publicViews/graphHeader.vue";
import GraphRenderer from "./publicViews/graphRenderer.vue";
import RankHeader from "./comparison/rankHeader.vue";
import RankRenderer from "./comparison/rankRenderer.vue";
import PolarHeader from "./comparison/polarHeader.vue";
import PolarRenderer from "./comparison/polarRenderer.vue";
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
import GraphTsneHeader from "./publicViews/graphTsneHeader.vue";
import GraphTsneRenderer from "./publicViews/graphTsneRenderer.vue";
import TopoLatentDensityRenderer from "./publicViews/topoLatentDensityRenderer.vue";
import TopoLatentDensityHeader from "./publicViews/topoLatentDensityHeader.vue";
import ScatterSymbolLegend from "./header/scatterSymbolLegend.vue";
import { isDbWiseComparativeDb } from "@/types/typeFuncs";

const props = defineProps({
    compDbId: {
        type: String,
        required: true,
        default: "",
    },
});

onMounted(() => {
    console.warn("compDb ", props.compDbId, "Mounted!");
});
onUnmounted(() => {
    console.warn("compDb ", props.compDbId, "Unmounted!");
});
const myStore = useMyStore();
const db = myStore.getCompDashboardById(props.compDbId)!;
const ds1 = myStore.getDatasetByName(db.refDatasetsNames[0] || "")!;
const ds2 = myStore.getDatasetByName(db.refDatasetsNames[1] || "")!;
// console.log( "in compDb: id:", props.compDbId, "\ndb is", db, "\nds1 is", ds1, "\nds2 is", ds2);

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION data on dashboard created
const calcSubCoords = async () => {
    const rootDb = myStore.compDashboardList.find((d) => d.isRoot);
    /////////////////////////////////////////
    ////// calc force-directed layout
    const {
        workerFn: graphWorkerFn,
        // workerStatus: graphWorkerStatus,
        workerTerminate: graphWorkerTerminate,
    } = useWebWorkerFn(calcGraphCoords, {
        timeout: 20_000,
        dependencies: [
            // "http://localhost:5173/workers/d3js.org_d3.v7.js",
            // "https://d3js.org/d3.v7.js", //for debug
            "https://d3js.org/d3.v7.min.js"
        ],
    });
    const localNodes =
        toRaw(ds1.nodes || ds2.nodes)?.filter((d) => db.srcNodesDict[d.id]) ||
        [];
    const localLinks =
        toRaw(ds1.links || ds2.links)?.filter(
            (d) => db.srcNodesDict[d.source] && db.srcNodesDict[d.target]
        ) || [];
    db.graphCoordsRet = await graphWorkerFn(localNodes, localLinks);
    db.srcLinksArr = localLinks;
    db.srcLinksDict = localLinks.reduce(
        (acc, cur) => ({
            ...acc,
            [cur.eid]: true,
        }),
        {}
    );
    console.log(
        "in compDb",
        props.compDbId,
        "in calcSubCoords, useWebWorker, got graphCoordsRet: ",
        db.graphCoordsRet
    );
    graphWorkerTerminate();
    ////// calc force-directed layout
    /////////////////////////////////////////

    db.tsneRet1 = rootDb?.tsneRet1.filter((d) => db.srcNodesDict[d.id]) || [];
    db.tsneRet2 = rootDb?.tsneRet2.filter((d) => db.srcNodesDict[d.id]) || [];

    if (rootDb?.graphTsneRet1) {
        const graphMapNodes = nodeMapGraph2GraphMapNodes(db.srcNodesDict);
        db.graphTsneRet1 = rootDb.graphTsneRet1.filter(
            (d) => graphMapNodes[d.id]
        );
    }

    if (rootDb?.graphTsneRet2) {
        const graphMapNodes = nodeMapGraph2GraphMapNodes(db.srcNodesDict);
        db.graphTsneRet2 = rootDb.graphTsneRet2.filter(
            (d) => graphMapNodes[d.id]
        );
    }
};

const loadDbData = async () => {
    console.log("load compDb Data!");
    if (db) {
        db.nodesSelections["full"] = JSON.parse(
            JSON.stringify(db.srcNodesDict)
        ); //{...undefined} = {}
        if (db.isRoot && db.isComplete) {
            // REVIEW nothing need?
            // console.log("in loadCompDbData, db.srcNodesDict", db.srcNodesDict);
            // console.log("in loadDbData, after assign db.nodesSelections", db.nodesSelections, "db.nodesSelections['full']", db.nodesSelections["full"]);
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
                    myStore
                        .getViewByName(db, "Latent Space - Model 1")
                        ?.setAttr("headerComp", shallowRef(TsneHeader))
                        .setAttr("bodyComp", shallowRef(TsneRenderer))
                        .setAttr("headerProps", { which: 1 })
                        .setAttr("bodyProps", { which: 1 })
                        .setAttr("initialWidth", 800);
                    myStore
                        .getViewByName(db, "Latent Space - Model 2")
                        ?.setAttr("headerComp", shallowRef(TsneHeader))
                        .setAttr("bodyComp", shallowRef(TsneRenderer))
                        .setAttr("headerProps", { which: 2 })
                        .setAttr("bodyProps", { which: 2 })
                        .setAttr("initialWidth", 800);
                    myStore
                        .getViewByName(db, "Topo + Latent Density - Model 1")
                        ?.setAttr(
                            "bodyComp",
                            shallowRef(TopoLatentDensityRenderer)
                        )
                        .setAttr("bodyProps", { which: 1 })
                        .setAttr(
                            "headerComp",
                            shallowRef(TopoLatentDensityHeader)
                        )
                        .setAttr("headerProps", { which: 1 });
                    myStore
                        .getViewByName(db, "Topo + Latent Density - Model 2")
                        ?.setAttr(
                            "bodyComp",
                            shallowRef(TopoLatentDensityRenderer)
                        )
                        .setAttr("bodyProps", { which: 2 })
                        .setAttr(
                            "headerComp",
                            shallowRef(TopoLatentDensityHeader)
                        )
                        .setAttr("headerProps", { which: 2 });
                    if (ds1.taskType === "node-classification") {
                        myStore
                            .getViewByName(db, "Topology Space")
                            ?.setAttr("bodyComp", shallowRef(GraphRenderer))
                            .setAttr("headerComp", shallowRef(GraphHeader));
                        myStore
                            .getViewByName(db, "Prediction Space - Model 1")
                            ?.setAttr(
                                "bodyComp",
                                shallowRef(ConfusionMatrixRenderer)
                            )
                            .setAttr("bodyProps", { which: 1 });
                        myStore
                            .getViewByName(db, "Prediction Space - Model 2")
                            ?.setAttr(
                                "bodyComp",
                                shallowRef(ConfusionMatrixRenderer)
                            )
                            .setAttr("bodyProps", { which: 2 });
                    } else if (ds1.taskType === "link-prediction") {
                        myStore
                            .getViewByName(db, "Topology Space")
                            ?.setAttr("bodyComp", shallowRef(GraphRenderer))
                            .setAttr("headerComp", shallowRef(GraphHeader));

                        myStore
                            .getViewByName(db, "Prediction Space - Model 1")
                            ?.setAttr("bodyComp", shallowRef(LinkPredRenderer))
                            .setAttr("bodyProps", { which: 1 })
                            .setAttr("headerComp", shallowRef(LinkPredHeader))
                            .setAttr("headerProps", { which: 1 });
                        myStore
                            .getViewByName(db, "Prediction Space - Model 2")
                            ?.setAttr("bodyComp", shallowRef(LinkPredRenderer))
                            .setAttr("bodyProps", { which: 2 })
                            .setAttr("headerComp", shallowRef(LinkPredHeader))
                            .setAttr("headerProps", { which: 2 });
                    } else if (ds1.taskType === "graph-classification") {
                        myStore
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
                        myStore
                            .getViewByName(db, "Prediction Space - Model 1")
                            ?.setAttr(
                                "bodyComp",
                                shallowRef(ConfusionMatrixRenderer)
                            )
                            .setAttr("bodyProps", { which: 1 });
                        myStore
                            .getViewByName(db, "Prediction Space - Model 2")
                            ?.setAttr(
                                "bodyComp",
                                shallowRef(ConfusionMatrixRenderer)
                            )
                            .setAttr("bodyProps", { which: 2 });

                        myStore
                            .getViewByName(db, "Graph Latent Space - Model 1")
                            ?.setAttr("headerComp", shallowRef(GraphTsneHeader))
                            .setAttr("bodyComp", shallowRef(GraphTsneRenderer))
                            .setAttr("headerProps", { which: 1 })
                            .setAttr("bodyProps", { which: 1 });
                        myStore
                            .getViewByName(db, "Graph Latent Space - Model 2")
                            ?.setAttr("headerComp", shallowRef(GraphTsneHeader))
                            .setAttr("bodyComp", shallowRef(GraphTsneRenderer))
                            .setAttr("headerProps", { which: 2 })
                            .setAttr("bodyProps", { which: 2 });
                    }
                    if (ds1.nodeSparseFeatureValues) {
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
                    if (ds1.denseNodeFeatures) {
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
                    resolve();
                })
                .catch((err) => {
                    console.error(
                        "in compDb, in loadDbData().then(), caught error",
                        err
                    );
                    reject(err);
                });
        });
    }
})(); //onCreated

/*
//NOTE - 暂时的用来debug的加载过程。实际上使用上面的函数：
// 即针对子view共享一个loading的Promise，而非将dashboard遮住
const isLoading = ref(true);
const error = ref(null);
const loadData = async () => {
    try {
        isLoading.value = true;
        error.value = null;
        await db.globalLoadPromise;
    } catch (e) {
        error.value = e;
    } finally {
        isLoading.value = false;
    }
};
loadData();
*/

////////////////// !SECTION data on dashboard created
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION data async (on selection, click, et al created
let rankView: RankView = {} as RankView,
    polarView: PolarView = {} as PolarView;
if (!isDbWiseComparativeDb(db)) {
    rankView = myStore.getViewByName(
        props.compDbId,
        "Comparative Rank View"
    ) as RankView; //将view抽出来的原因是：watch计算源和算法种类时，获取算法种类的computed包含了getViewByName，这样当viewList变更时，也会触发watch
    polarView = myStore.getViewByName(
        props.compDbId,
        "Comparative Polar View"
    ) as PolarView;
}

const RankComp = shallowRef<any>(null); //如果不想中间再跨一个到达svg，那就用这个
const rankProps = ref({});
const PolarComp = shallowRef<any>(null); //如果不想中间再跨一个到达svg，那就用这个//REVIEW :idle comp?
const polarProps = ref({});

const calcRank3Error = ref<ErrorEvent | Error>();

const comparativeSingleSel = computed(
    () => db.nodesSelections["comparativeSingle"] || {}
);
const comparativeMultiSel = computed(
    () => db.nodesSelections["comparative"] || {}
);
const rankEmbDiffAlgo = computed(() => rankView?.rankEmbDiffAlgo);

const doCalcRank3 = async (
    [newAlgo, newId, newDict]: [
        Type_RankEmbDiffAlgos | undefined,
        Record<Type_NodeId, any>, //实际运行中newId往往是string即只有键
        Record<Type_NodeId, any>
    ],
    [oldAlgo, oldId, oldDict]: [
        Type_RankEmbDiffAlgos | undefined,
        Record<Type_NodeId, any>,
        Record<Type_NodeId, any>
    ] = [undefined, {}, {}]
) => {
    if (isDbWiseComparativeDb(db)) return;
    console.log(
        "in doCalcRank3, new and old: ",
        [newAlgo, newId, newDict],
        [oldAlgo, oldId, oldDict]
    );
    if (db.rankWorker.workerStatus === "RUNNING") {
        console.log("new calcRank3 encountered!");
        db.rankWorker.workerTerminate();
    }
    calcRank3Error.value = undefined;
    try {
        rankView.rankData = await db.rankWorker.workerFn(
            newAlgo,
            newAlgo === "single"
                ? toRaw(comparativeSingleSel.value)
                : toRaw(comparativeMultiSel.value),
            toRaw(ds1.embNode),
            toRaw(ds2.embNode)
        );
    } catch (e) {
        //reject
        calcRank3Error.value = e as Error | ErrorEvent;
        console.error("now in doCalcRank3 we catch an error", e);
    } finally {
        db.rankWorker.workerTerminate();
    }
};
watch(
    [
        // rankEmbDiffAlgo,
        // comparativeSingleSel,
        // comparativeMultiSel,
        () => rankView.rankEmbDiffAlgo, //一个意思
        () => db.nodesSelections["comparativeSingle"],
        () => db.nodesSelections["comparative"],
    ],
    doCalcRank3,
    {
        deep: true,
        immediate: false,
        // onTrigger(event) {//NOTE debug, only in dev mode
        //     console.warn("watch rank calc trigger!", event);
        // },
    }
);

watch(
    () => db.rankWorker.workerStatus,
    (newV) => {
        if (newV === "PENDING") {
            rankProps.value = { text: "waiting for nodes selection" };
            RankComp.value = PendingComp;
        } else if (newV === "RUNNING") {
            rankProps.value = { text: "calculating rank" };
            RankComp.value = LoadingComp;
        } else if (newV === "SUCCESS") {
            // rankProps.value = { data: calcRank3Ret.value };
            RankComp.value = RankRenderer;
            // defineAsyncComponent({
            // loader: async () => {
            //     return await import("./comparison/rankRenderer.vue");
            // },
            // delay: 200,
            // timeout: 1000,
            //     loadingComponent: LoadingComp, //NOTE 此时的loading就是网络传输rankRenderer.vue文件的loading了
            //     errorComponent: ErrorComp, //NOTE 此时的error就是网络传输rankRenderer.vue文件的error了
            // });
        } else {
            rankProps.value = { error: calcRank3Error.value?.message };
            RankComp.value = h(
                "div",
                {
                    style: { width: "80%", margin: "10% auto" },
                },
                [
                    //懒得再写一个组件了，直接上渲染函数
                    h(ErrorComp, rankProps.value),
                    h(
                        ElButton,
                        {
                            onClick: (e: Event) => {
                                doCalcRank3([
                                    rankEmbDiffAlgo.value,
                                    comparativeSingleSel.value,
                                    comparativeMultiSel.value,
                                ]);
                                myStore.repairButtonFocus(e);
                            },
                        },
                        () => "retry" //Non-function value encountered for default slot. Prefer function slots for better performance.
                    ),
                ]
            );
        }
    },
    { immediate: true }
);

const calcPolarError = ref<ErrorEvent | Error>();

const polarEmbDiffAlgo = computed(() => polarView.polarEmbDiffAlgo);
const polarTopoDistAlgo = computed(() => polarView.polarTopoDistAlgo);
const nodeMapLink = ds1.nodeMapLink || ds2.nodeMapLink || [];
const neighborMasksByHop =
    ds1.neighborMasksByHop || ds2.neighborMasksByHop || [];
const hops = ds1.hops || ds2.hops || myStore.defaultHops;

const doCalcPolar = async (
    [newEmbAlgo, newTopoAlgo, newId, newDict]: [
        Type_polarEmbDiffAlgos | undefined,
        Type_polarTopoDistAlgos | undefined,
        Record<Type_NodeId, any>, //NOTE 实际运行中newId往往是string即只有键
        Record<Type_NodeId, any>
    ],
    [oldEmbAlgo, oldTopoAlgo, oldId, oldDict]: [
        Type_polarEmbDiffAlgos | undefined,
        Type_polarTopoDistAlgos | undefined,
        Record<Type_NodeId, any>,
        Record<Type_NodeId, any>
    ] = [undefined, undefined, {}, {}]
) => {
    if (isDbWiseComparativeDb(db)) return;
    console.log(
        "in doCalcPolar, new and old: ",
        [newEmbAlgo, newTopoAlgo, newId, newDict],
        [oldEmbAlgo, oldTopoAlgo, oldId, oldDict]
    );
    if (db.polarWorker.workerStatus === "RUNNING") {
        console.log("new calcPolar encountered!");
        db.polarWorker.workerTerminate();
    }
    calcPolarError.value = undefined;
    try {
        polarView.polarData = await db.polarWorker.workerFn(
            newTopoAlgo,
            newEmbAlgo,
            newEmbAlgo === "single"
                ? toRaw(comparativeSingleSel.value)
                : toRaw(comparativeMultiSel.value),
            toRaw(ds1.embNode),
            toRaw(ds2.embNode),
            toRaw(nodeMapLink),
            hops,
            toRaw(neighborMasksByHop)
        );
        polarView.hops = hops;
    } catch (e) {
        //reject
        calcPolarError.value = e as Error | ErrorEvent;
        console.error("now in doCalcPolar we catch an error", e);
    } finally {
        db.polarWorker.workerTerminate();
    }
};
watch(
    [
        polarEmbDiffAlgo,
        polarTopoDistAlgo,
        comparativeSingleSel,
        comparativeMultiSel,
    ],
    doCalcPolar,
    {
        deep: true,
        immediate: false,
        // onTrigger(event) {//NOTE debug, only in dev mode
        //     console.warn("watch polar calc trigger!", event);
        // },
    }
);

watch(
    () => db.polarWorker.workerStatus,
    (newV) => {
        if (newV === "PENDING") {
            polarProps.value = { text: "waiting for nodes selection" };
            PolarComp.value = PendingComp;
        } else if (newV === "RUNNING") {
            polarProps.value = { text: "calculating polar view data" };
            PolarComp.value = LoadingComp;
        } else if (newV === "SUCCESS") {
            // polarProps.value = { data: calcPolarRet.value, hops: hops };
            PolarComp.value = PolarRenderer;
            // defineAsyncComponent({
            //     loader: async () => {
            //         return await import("./comparison/polarRenderer.vue");
            //     },
            //     delay: 200,
            //     timeout: 1000,
            //     errorComponent: ErrorComp, //NOTE 此时的error就是网络传输polarRenderer.vue文件的error了
            //     loadingComponent: LoadingComp, //NOTE 此时的loading就是网络传输polarRenderer.vue文件的loading了
            // });
        } else {
            polarProps.value = { error: calcPolarError.value?.message };
            PolarComp.value = h(
                "div",
                {
                    style: { width: "80%", margin: "10% auto" },
                },
                [
                    //懒得再写一个组件了，直接上渲染函数

                    h(ErrorComp, polarProps.value),
                    h(
                        ElButton,
                        {
                            onClick: (e: Event) => {
                                doCalcPolar([
                                    polarEmbDiffAlgo.value,
                                    polarTopoDistAlgo.value,
                                    comparativeSingleSel.value,
                                    comparativeMultiSel.value,
                                ]);
                                myStore.repairButtonFocus(e);
                            },
                        },
                        () => "retry" //Non-function value encountered for default slot. Prefer function slots for better performance.
                    ),
                ]
            );
        }
    },
    { immediate: true }
);
////////////////// !SECTION data async (on selection, click, et al created
////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////
// STUB : 一个dashboard中有哪些特征、views，
if (!isDbWiseComparativeDb(db)) {
    rankView
        .setAttr("headerComp", shallowRef(RankHeader))
        .setAttr("bodyComp", RankComp)
        .setAttr("bodyProps", rankProps)
        .setAttr("initialWidth", 800);
    polarView
        .setAttr("headerComp", shallowRef(PolarHeader))
        .setAttr("bodyComp", PolarComp)
        .setAttr("bodyProps", polarProps)
        .setAttr("initialWidth", 800);
}
// END STUB
/////////////////////////////////////////////////////////////////////////////////////////////

//affix悬浮效果相关
const affixOffset = computed(() => window?.innerHeight * 0.05 || 50);

// draggable相关
const dragOptions = ref({
    animation: 500,
    // group: "description",
    disabled: false,
    ghostClass: "ghost",
});
const isDragging = ref(false);

//选择相关
const handleFilter = async (e: Event) => {
    const selAndNeighbor = calcNeighborDict(
        db.nodesSelections["public"],
        ds1.hops || ds2.hops!,
        ds1.neighborMasksByHop || ds2.neighborMasksByHop!
    );

    const ret: Record<
        Type_NodeId,
        { gid: Type_GraphId; parentDbIndex: number; hop: number }
    > = {};
    for (const id in selAndNeighbor) {
        ret[id] = {
            ...selAndNeighbor[id],
            parentDbIndex: 0,
            gid: db.srcNodesDict[id].gid,
        };
    }

    await myStore.calcPrincipalSnapshotOfDashboard(db);
    myStore.addCompDashboard(
        {
            id: nanoid(),
            refDatasetsNames: [...db.refDatasetsNames],
            date: Date.now ? Date.now() : new Date().getTime(),
            isRoot: false,
            isComplete: false,
            parentId: db.id,
            fromViewName: db.fromViewName,
            graphCoordsRet: undefined,
            srcNodesDict: {
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
    // UIStore.repairButtonFocus(e);
    myStore.repairButtonFocus(e);
};

// represent related
const handleRepresent = async (e: Event | undefined) => {
    //assume len >=2
    // const { id } = myStore.recentCompDashboardList.pop()!;
    // myStore.representedCompDashboardList.push(id);

    await myStore.calcPrincipalSnapshotOfDashboard(props.compDbId);
    db.isRepresented = true;
    const { id } = myStore.recentCompDashboardList.at(-2)!;
    myStore.toCompDashboardById(id);
    // myStore.recentCompDashboardList.at(-1)!.isRepresented = true;
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
    resizableBoxRef,
    (newV) => {
        if (newV) {
            // const { width: principalViewWidth, height: principalViewHeight } =
            //     newV;
            console.log(
                "in db",
                db.name,
                "watch resizableBoxRef changed to",
                newV
            );
        }
    },
    { immediate: true }
);
// const draggableContainerRef = ref<InstanceType<typeof draggable> | null>(null);
// watch(
//     draggableContainerRef,
//     (newV) => {
//         console.log(
//             "in db",
//             db.name,
//             "watch draggableContainerRef changed to",
//             newV
//         );
//     },
//     { immediate: true }
// );

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
    // myStore.toCompDashboardById(props.compDbId);
    db.isRepresented = false;
    myStore.toCompDashboardById(props.compDbId);
    myStore.repairButtonFocus(e);
};
const handleMinimize = (e: Event) => {
    db.isRepresented = false;
    const i = myStore.recentCompDashboardList.findIndex(
        (d) => d.id === props.compDbId
    );

    if (i < 0) {
        throw new Error(`in representedDb failed to find ${props.compDbId}!`);
    } else {
        if (i < myStore.recentCompDashboardList.length - myStore.recentLen) {
            //说明renderList里面没有，应当更新renderIndex
            myStore.recentCompDashboardList[i].renderIndex =
                myStore.recentCompDashboardList.at(
                    -myStore.recentLen
                )!.renderIndex;
            myStore.recentCompDashboardList.at(
                -myStore.recentLen
            )!.renderIndex = 0;
        }
        const to = myStore.recentCompDashboardList.splice(i, 1); //这里是个数组;
        // if (to[0].isRepresented) to[0].isRepresented = false;

        myStore.recentCompDashboardList.splice(-1, 0, to[0]); //插到倒数第二个，代表优先级次优
    }
    myStore.repairButtonFocus(e);
};
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

.represented-comp-dashboard-head {
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

.represented-comp-dashboard-body {
    margin: 0;
    padding: 0;
    opacity: 1;
    background-color: white;
    height: 92%;
}

.represented-comp-dashboard {
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

.comp-dashboard {
    /* width: 100%; */
    /* height: auto; */
    /* NOTE 其实flex会自动计算高度。不用自己算。*/
    padding: 0;
    margin: 0;
    /* position: relative; */
    border: 1px solid rgba(0, 0, 0, 0.16);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.16), 0 0 6px rgba(0, 0, 0, 0.08);
}

.comp-dashboard-head {
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
    /* 此原本是comp-dashboard-content，加入draggable，多了一层*/
    /* width: 100%; */
    /* height: v-bind("height"); */
    /* 不用计算也可以。费那个劲呢。*/
    margin: 0;
    padding-bottom: 20px;
    padding-right: 20px;
    /* 当不设置width时，padding的bug就好了 */

    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    justify-content: flex-start;
    align-items: start;
}

.draggable-container > * {
    margin-top: 20px;
    margin-left: 20px;
    /* border: 1px solid black; */
}

.cursor-move {
    cursor: move;
}
</style>
