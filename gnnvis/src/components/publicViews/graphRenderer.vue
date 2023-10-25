<template>
    <svg
        ref="svgRef"
        :width="view.bodyWidth"
        :height="view.bodyHeight"
        version="1.1"
        xmlns="http://www.w3.org/2000/svg"
        baseProfile="full"
    >
        <g
            id="globalTransform"
            :transform="`translate(${transformRef.x},${transformRef.y}) scale(${transformRef.k})`"
        >
            <g
                id="links"
                v-if="view.isShowLinks"
                stroke="#999"
                :stroke-opacity="view.linkOpacity"
                stroke-linecap="round"
            >
                <line
                    v-for="(l, i) in links"
                    :key="i"
                    :stroke-width="1"
                    :stroke-opacity="
                        db.isHighlightCorrespondingNode &&
                        !isEmptyDict(db.highlightedNodeIds)
                            ? 0.1
                            : view.linkOpacity
                    "
                    :x1="renderCoords[mapNodeId2Index[l.source]].x"
                    :y1="renderCoords[mapNodeId2Index[l.source]].y"
                    :x2="renderCoords[mapNodeId2Index[l.target]].x"
                    :y2="renderCoords[mapNodeId2Index[l.target]].y"
                ></line>
            </g>
            <g v-if="view.isShowHopSymbols">
                <template v-for="n in renderCoords" :key="n.id">
                    <g :transform="`translate(${n.x} ${n.y})`">
                        <path
                            class="outer"
                            :d="getSymbolPathByNodeId(n.id)!"
                            :fill="
                                isDbWiseComparativeDb(db)
                                    ? 'white'
                                    : colorScale(labels[+n.id] + '')
                            "
                            :stroke="
                                renderEntry[n.id]
                                    ? 'black'
                                    : colorScale(labels[+n.id] + '')
                            "
                            :stroke-width="1"
                            :opacity="dynamicOpacity(n.id)"
                            @mouseenter="
                                db.highlightedNodeIds[n.id] =
                                    db.srcNodesDict[n.id]
                            "
                            @mouseleave="db.highlightedNodeIds = {}"
                        >
                            <title>
                                {{ `id: ${n.id}\ntrueLabel: ${labels[+n.id]}` }}
                            </title>
                        </path>
                        <path
                            v-if="isDbWiseComparativeDb(db)"
                            class="LR"
                            :d="getLRSymbolPathByNodeId(n.id)!"
                            stroke-linejoin="bevel"
                            :fill="colorScale(labels[+n.id] + '')"
                            :stroke="
                                renderEntry[n.id]
                                    ? 'black'
                                    : colorScale(labels[+n.id] + '')
                            "
                            :stroke-width="1"
                            :opacity="dynamicOpacity(n.id)"
                            @mouseenter="
                                db.highlightedNodeIds[n.id] =
                                    db.srcNodesDict[n.id]
                            "
                            @mouseleave="db.highlightedNodeIds = {}"
                        >
                            <title>
                                {{ `id: ${n.id}\ntrueLabel: ${labels[+n.id]}` }}
                            </title>
                        </path>
                    </g>
                </template>
            </g>
            <g
                v-else
                id="circles"
                stroke="none"
                stroke-opacity="1"
                stroke-width="1"
            >
                <circle
                    v-for="n in renderCoords"
                    :key="n.id"
                    :r="view.nodeRadius"
                    :fill="colorScale(labels[+n.id] + '')"
                    :cx="n.x"
                    :cy="n.y"
                    :stroke="renderEntry[n.id] ? 'black' : 'none'"
                    :opacity="dynamicOpacity(n.id)"
                    @mouseenter="
                        db.highlightedNodeIds[n.id] = db.srcNodesDict[n.id]
                    "
                    @mouseleave="db.highlightedNodeIds = {}"
                >
                    <title>
                        {{ `id: ${n.id}\ntrueLabel: ${labels[+n.id]}` }}
                    </title>
                </circle>
            </g>
            <g class="gBrush" ref="brushRef"></g>
        </g>
    </svg>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from "vue";
import { isEmptyDict, rescaleCoords } from "../../utils/graphUtils";
import { useD3Brush } from "../plugin/useD3Brush";
import { useD3Zoom } from "../plugin/useD3Zoom";
import { useResize } from "../plugin/useResize";
import { useMyStore } from "@/stores/store";
import type {
    CompDashboard,
    LinkableView,
    Node,
    NodeCoord,
    NodeView,
    SingleDashboard,
    Type_NodeId,
    Type_NodesSelectionEntryId,
} from "@/types/types";
import {
    circlePath,
    getAreaBySymbolOuterRadius,
    leftHalfAndWholeCirclePathStroke,
    leftHalfCirclePath,
    rightHalfAndWholeCirclePathStroke,
    rightHalfCirclePath,
} from "@/utils/otherUtils";
import * as d3 from "d3";
import { isDbWiseComparativeDb } from "@/types/typeFuncs";

const props = defineProps({
    dbId: {
        type: String,
        default: "",
    },
    viewName: {
        type: String,
        required: true,
    },
});

const svgRef = ref<SVGElement | null>(null);
const myStore = useMyStore();

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION data prepare
const db = myStore.getTypeReducedDashboardById(props.dbId)!;
const ds = Object.hasOwn(db, "refDatasetsNames")
    ? myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[0]) ||
      myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[1])
    : myStore.getDatasetByName((db as SingleDashboard).refDatasetName);
const view = myStore.getViewByName(db, props.viewName) as NodeView<
    Node & NodeCoord & d3.SimulationNodeDatum
> &
    LinkableView;
const nodes = computed(
    () =>
        (db.graphCoordsRet as (Node & d3.SimulationNodeDatum & NodeCoord)[]) ||
        []
);
const mapNodeId2Index = computed<Record<Type_NodeId, number>>(() =>
    nodes.value.reduce(
        (acc, cur, curIndex) => ({
            ...acc,
            [cur.id]: curIndex,
        }),
        {}
    )
);
const trueLabels = ds.trueLabels || [];
const predLabels = ds.predLabels || [];
const labels = computed(() =>
    db.labelType === "true" ? trueLabels : predLabels
);
const links = computed(() => db.srcLinksArr || []);
const colorScale = ds.colorScale || (() => "#888");
const dynamicOpacity = computed(
    () => (i: string) =>
        !db.isHighlightCorrespondingNode || isEmptyDict(db.highlightedNodeIds)
            ? 1
            : db.highlightedNodeIds[i]
            ? 1
            : 0.1
);
////////////////// !SECTION data prepare
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION nodeSelectionEntry related
const srcEntryIds = myStore.getViewSourceNodesSelectionEntry(props.viewName);
const tgtEntryIds = myStore.getViewTargetNodesSelectionEntry(props.viewName);

const renderEntry = computed(() => db.nodesSelections[tgtEntryIds[0]]);

const curSrcEntryId = ref<Type_NodesSelectionEntryId>(srcEntryIds[0]);
const srcNodesDict = computed(() => db.nodesSelections[curSrcEntryId.value]);
// console.log( "in ", props.viewName, " srcEntryIds", srcEntryIds, "\ntgtEntryIds", tgtEntryIds, "\nsrcNodeDict", srcNodesDict.value);
const getSymbolPathByNodeId = (id: Type_NodeId) => {
    if (srcNodesDict.value[id].hop === -1) {
        return circlePath(view.nodeRadius);
    } else {
        const { hop: hopIndex } = srcNodesDict.value[id];
        return myStore.symbolPathByHopAndParentIndex[hopIndex][2](
            view.nodeRadius
        );
    }
};
const getLRSymbolPathByNodeId = (id: Type_NodeId) => {
    if (srcNodesDict.value[id].hop === -1) {
        switch (srcNodesDict.value[id].parentDbIndex) {
            case 0:
                return leftHalfCirclePath(view.nodeRadius);
            case 1:
                return rightHalfCirclePath(view.nodeRadius);
            case 2:
                return circlePath(view.nodeRadius);
        }
    } else {
        const { hop: hopIndex, parentDbIndex } = srcNodesDict.value[id];
        return myStore.symbolPathByHopAndParentIndex[hopIndex][parentDbIndex](
            view.nodeRadius
        );
    }
};
////////////////// !SECTION nodeSelectionEntry related
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION points first calc
watch(
    [srcNodesDict, nodes],
    ([newDict, newNodes]) => {
        view.sourceCoords = newNodes.filter(
            (d, i) => newDict[d.id]
            //NOTE这里不能用i做索引！
        );
    },
    { immediate: true, deep: true }
);
watch(
    () => view.sourceCoords,
    (newV) => {
        console.log(
            "in db",
            db.name,
            "in graph, view.sourceCoords changed to",
            newV,
            "\nlinks is, db.srcLinksArr",
            links.value
        );
        view.rescaledCoords = rescaleCoords<
            NodeCoord & Node & d3.SimulationNodeDatum,
            never
        >(
            newV || [],
            [
                view.bodyMargins.left * view.bodyWidth,
                view.bodyWidth * (1 - view.bodyMargins.right),
            ],
            [
                view.bodyMargins.top * view.bodyHeight,
                view.bodyHeight * (1 - view.bodyMargins.bottom),
            ],
            (d) => d.x,
            (d) => d.y,
            db.name + "===" + props.viewName
        ) as (NodeCoord & Node & d3.SimulationNodeDatum)[];
    },
    {
        immediate: true,
        deep: true,
    }
);
////////////////// !SECTION points first calc
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION resize
//NOTE resize必须在brush和zoom之前，而brush和zoom谁前后无所谓
const { widthScaleRatio, heightScaleRatio } = useResize(
    () => view.resizeEndSignal,
    () => db.whenToRescaleMode === "onResizeEnd",
    // view.rescaledCoords,
    () => view.bodyWidth,
    () => view.bodyHeight,
    () => {
        console.log(
            "in",
            props.dbId + props.viewName,
            "rescale Caller called!"
        );
        view.rescaledCoords = rescaleCoords<
            NodeCoord & Node & d3.SimulationNodeDatum,
            never
        >(
            view.sourceCoords,
            [
                view.bodyMargins.left * view.bodyWidth,
                view.bodyWidth * (1 - view.bodyMargins.right),
            ],
            [
                view.bodyMargins.top * view.bodyHeight,
                view.bodyHeight * (1 - view.bodyMargins.bottom),
            ],
            (d: Node & NodeCoord & d3.SimulationNodeDatum) => d.x,
            (d: Node & NodeCoord & d3.SimulationNodeDatum) => d.y,
            db.name + "===" + props.viewName
        ) as (NodeCoord & Node & d3.SimulationNodeDatum)[];
    },
    db.name + "===" + props.viewName
);
const renderCoords = computed(() => view.rescaledCoords);
////////////////// !SECTION resize
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION brush
const brushRef = ref<SVGGElement | null>(null);
const { enableBrushFuncRef, disableBrushFuncRef, hideBrushRectFuncRef } =
    useD3Brush(
        brushRef,
        () => [
            [0, 0],
            [view.bodyWidth, view.bodyHeight],
        ],
        () => [
            [
                view.bodyMargins.left * view.bodyWidth,
                view.bodyMargins.top * view.bodyHeight,
            ],
            [
                view.bodyWidth * (1 - view.bodyMargins.right),
                view.bodyHeight * (1 - view.bodyMargins.bottom),
            ],
        ],
        renderCoords,
        () => {
            // localTargetEntry.value = {};
            tgtEntryIds.forEach((entryId) => {
                db.nodesSelections[entryId] = {};
            });
        },
        // localTargetEntry,
        (id: Type_NodeId) => {
            db.fromViewName = props.viewName;
            tgtEntryIds.forEach((entryId) => {
                db.nodesSelections[entryId][id] = db.srcNodesDict[id];
            });
        },
        () => true,
        () => db.clearSelMode === "manual",
        view.isBrushEnabled,
        (d) => d.x,
        (d) => d.y,
        widthScaleRatio,
        heightScaleRatio,
        db.name + "===" + props.viewName
    );
view.brushEnableFunc = enableBrushFuncRef;
view.brushDisableFunc = disableBrushFuncRef;
view.hideRectWhenClearSelFunc = hideBrushRectFuncRef;
////////////////// !SECTION brush
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION zoom
const { transformRef, enablePanFuncRef, disablePanFuncRef, resetZoomFuncRef } =
    useD3Zoom(
        svgRef,
        undefined,
        () => view.bodyWidth,
        () => view.bodyHeight,
        (w, h) => [
            [0, 0],
            [w, h],
        ],
        undefined,
        undefined,
        !view.isBrushEnabled,
        db.name + "===" + props.viewName
    );
view.panEnableFunc = enablePanFuncRef;
view.panDisableFunc = disablePanFuncRef;
view.resetZoomFunc = resetZoomFuncRef;
////////////////// !SECTION zoom
////////////////////////////////////////////////////////////////////////////////

onMounted(() => {
    console.log("graph renderer, mounted!");
}); // onMounted END
onUnmounted(() => {
    console.log("graph renderer, unmounted!");
});
</script>

<style scoped></style>
