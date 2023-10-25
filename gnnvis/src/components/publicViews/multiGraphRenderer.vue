<template>
    <svg :height="view.bodyHeight * 0.04" :width="view.bodyWidth">
        <g :transform="`translate(${mL}, ${view.bodyHeight * 0.03})`">
            <text :font-size="view.bodyHeight * 0.03">
                inner rect: pred graph label, outer rect: true graph label
            </text>
        </g>
    </svg>
    <el-scrollbar :height="view.bodyHeight * 0.96">
        <svg
            ref="svgRef"
            :width="view.bodyWidth"
            :height="chartHeight"
            version="1.1"
            xmlns="http://www.w3.org/2000/svg"
            baseProfile="full"
        >
            <g
                class="globalTransform"
                :transform="`translate(${transformRef.x},${transformRef.y}) scale(${transformRef.k})`"
            >
                <g
                    class="margin"
                    :transform="`translate(${mL},${mT})`"
                    :width="contentWidth"
                    :height="contentHeight"
                >
                    <g
                        v-for="(graph, i) in view.groupedRescaledCoords"
                        :key="graph.gid"
                        :transform="`translate(${
                            (gap + subWidth) * (i % numColumns)
                        },${
                            Math.floor(i / numColumns) * (gap + view.subHeight)
                        })`"
                        class="singleGraph"
                    >
                        <rect
                            class="outerBorder"
                            :x="1"
                            :y="1"
                            :width="Math.max(0, subWidth - 2)"
                            :height="Math.max(0, view.subHeight - 2)"
                            :stroke="
                                graphColorScale(
                                    graphTrueLabels[+graph.gid] + ''
                                )
                            "
                            :stroke-width="2"
                            fill="white"
                            :fill-opacity="0"
                        >
                            <title>
                                {{
                                    `graphId: ${graph.gid}\ngraphPredLabel: ${
                                        graphPredLabels[+graph.gid]
                                    }\ngraphTrueLabel: ${
                                        graphTrueLabels[+graph.gid]
                                    }`
                                }}
                            </title>
                        </rect>
                        <rect
                            v-if="isSingleDb(db)"
                            class="innerBorder"
                            :x="subInnerMarginLeft"
                            :y="subInnerMarginTop"
                            :width="subInnerWidth"
                            :height="subInnerHeight"
                            :stroke="
                                graphColorScale(
                                    graphPredLabels[+graph.gid] + ''
                                )
                            "
                            :stroke-width="2"
                            fill="white"
                            :fill-opacity="0"
                        >
                            <title>
                                {{
                                    `graphId: ${graph.gid}
                                    \ngraphPredLabel: ${
                                        graphPredLabels[+graph.gid]
                                    }
                                    \ngraphTrueLabel: ${
                                        graphTrueLabels[+graph.gid]
                                    }`
                                }}
                            </title>
                        </rect>
                        <g
                            v-else
                            class="innerBorder"
                            :stroke-width="2"
                            fill="white"
                            :fill-opacity="0"
                        >
                            <path
                                :d="`M${
                                    subInnerMarginLeft + subInnerWidth / 2
                                } ${subInnerMarginTop} H${subInnerMarginLeft} V${subInnerHeight} H${
                                    subInnerMarginLeft + subInnerWidth / 2
                                }`"
                                :stroke="
                                    graphColorScale(
                                        graphPredLabels[+graph.gid] + ''
                                    )
                                "
                            >
                                {{
                                    `graphId: ${graph.gid}
                                    \ndb0-graphPredLabel: ${
                                        graphPredLabels[+graph.gid]
                                    }
                                    \ngraphTrueLabel: ${
                                        graphTrueLabels[+graph.gid]
                                    }`
                                }}
                            </path>
                            <path
                                :d="`M${
                                    subInnerMarginLeft + subInnerWidth / 2
                                } ${subInnerMarginTop} H${
                                    subInnerMarginLeft + subInnerWidth
                                } V${subInnerHeight} H${
                                    subInnerMarginLeft + subInnerWidth / 2
                                }`"
                                :stroke="
                                    graphColorScale(
                                        graphOtherPredLabels[+graph.gid] + ''
                                    )
                                "
                            >
                                {{
                                    `graphId: ${graph.gid}
                                    \ndb1-graphPredLabel: ${
                                        graphOtherPredLabels[+graph.gid]
                                    }
                                    \ngraphTrueLabel: ${
                                        graphTrueLabels[+graph.gid]
                                    }`
                                }}
                            </path>
                        </g>
                        <g
                            class="links"
                            v-if="view.isShowLinks"
                            stroke="#555"
                            :stroke-opacity="view.linkOpacity"
                            stroke-linecap="round"
                            :transform="`translate(${subInnerMarginLeft},${subInnerMarginTop})`"
                        >
                            <line
                                v-for="linkId in graph.filteredLinks"
                                :key="linkId"
                                :stroke-width="1"
                                :stroke-opacity="
                                    db.isHighlightCorrespondingNode &&
                                    !isEmptyDict(db.highlightedNodeIds)
                                        ? 0.1
                                        : view.linkOpacity
                                "
                                :x1="
                                    graph.filteredNodes[
                                        mapGroupedNodeId2Index[
                                            localLinksArr[
                                                mapLinkId2Index[linkId]
                                            ].source
                                        ]
                                    ].x
                                "
                                :y1="
                                    graph.filteredNodes[
                                        mapGroupedNodeId2Index[
                                            localLinksArr[
                                                mapLinkId2Index[linkId]
                                            ].source
                                        ]
                                    ].y
                                "
                                :x2="
                                    graph.filteredNodes[
                                        mapGroupedNodeId2Index[
                                            localLinksArr[
                                                mapLinkId2Index[linkId]
                                            ].target
                                        ]
                                    ].x
                                "
                                :y2="
                                    graph.filteredNodes[
                                        mapGroupedNodeId2Index[
                                            localLinksArr[
                                                mapLinkId2Index[linkId]
                                            ].target
                                        ]
                                    ].y
                                "
                            ></line>
                        </g>
                        <g
                            v-if="view.isShowHopSymbols"
                            :transform="`translate(${subInnerMarginLeft},${subInnerMarginTop})`"
                        >
                            <template
                                v-for="n in graph.filteredNodes"
                                :key="n.id"
                            >
                                <g :transform="`translate(${n.x} ${n.y})`">
                                    <path
                                        class="outer"
                                        :d="getSymbolPathByNodeId(n.id)!"
                                        :fill="
                                            isDbWiseComparativeDb(db)
                                                ? 'white'
                                                : nodeColorScale(
                                                      nodeLabels[+n.id] + ''
                                                  )
                                        "
                                        :stroke="
                                            renderEntry[n.id]
                                                ? 'black'
                                                : nodeColorScale(
                                                      nodeLabels[+n.id] + ''
                                                  )
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
                                            {{
                                                `id: ${n.id}\ntrueLabel: ${
                                                    nodeLabels[+n.id]
                                                }`
                                            }}
                                        </title>
                                    </path>
                                    <path
                                        v-if="isDbWiseComparativeDb(db)"
                                        class="LR"
                                        :d="getLRSymbolPathByNodeId(n.id)!"
                                        stroke-linejoin="bevel"
                                        :fill="
                                            nodeColorScale(
                                                nodeLabels[+n.id] + ''
                                            )
                                        "
                                        :stroke="
                                            renderEntry[n.id]
                                                ? 'black'
                                                : nodeColorScale(
                                                      nodeLabels[+n.id] + ''
                                                  )
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
                                            {{
                                                `id: ${n.id}\ntrueLabel: ${
                                                    nodeLabels[+n.id]
                                                }`
                                            }}
                                        </title>
                                    </path>
                                </g>
                            </template>
                        </g>
                        <g
                            v-else
                            class="points"
                            stroke="none"
                            stroke-opacity="1"
                            stroke-width="1"
                            :transform="`translate(${subInnerMarginLeft},${subInnerMarginTop})`"
                        >
                            <circle
                                v-for="n in graph.filteredNodes"
                                :key="n.id"
                                :r="2"
                                :fill="nodeColorScale(nodeLabels[+n.id] + '')"
                                :cx="n.x"
                                :cy="n.y"
                                :stroke="renderEntry[n.id] ? 'black' : 'none'"
                                :opacity="dynamicOpacity(n.id)"
                                @mouseenter="
                                    db.highlightedNodeIds[n.id] =
                                        db.srcNodesDict[n.id]
                                "
                                @mouseleave="db.highlightedNodeIds = {}"
                            >
                                <title>
                                    {{
                                        `id: ${n.id}\ntrueLabel: ${
                                            nodeLabels[+n.id]
                                        }`
                                    }}
                                </title>
                            </circle>
                        </g>
                    </g>
                </g>
                <g class="gBrush" ref="brushRef"></g>
            </g>
        </svg>
    </el-scrollbar>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from "vue";
import {
    isEmptyDict,
    nodeMapGraph2GraphMapNodes,
    rescaleCoords,
} from "../../utils/graphUtils";
import { useD3Brush } from "../plugin/useD3Brush";
import { useD3Zoom } from "../plugin/useD3Zoom";
import { useResize } from "../plugin/useResize";
import { useMyStore } from "@/stores/store";
import type {
    CompDashboard,
    LinkableView,
    MultiGraphView,
    NodeCoord,
    SingleDashboard,
    Type_GraphId,
    Type_NodeId,
    Type_NodesSelectionEntryId,
} from "@/types/types";
import { isCompDb, isDbWiseComparativeDb, isSingleDb } from "@/types/typeFuncs";
import {
    circlePath,
    leftHalfCirclePath,
    rectPath,
    rightHalfCirclePath,
} from "@/utils/otherUtils";
defineOptions({
    //因为我们没有用一个整体的div包起来，需要禁止透传
    inheritAttrs: false, //NOTE vue 3.3+
});
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
const theOtherDs = Object.hasOwn(db, "refDatasetsNames")
    ? myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[1]) ||
      myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[0])
    : myStore.getDatasetByName((db as SingleDashboard).refDatasetName);

const view = myStore.getViewByName(db, props.viewName) as MultiGraphView;

const nodeLabels = computed(() => ds?.trueLabels || []);
const nodeColorScale = ds.colorScale || (() => "#888");
const graphTrueLabels = computed(() => ds.graphTrueLabels || []);
const graphPredLabels = computed(() => ds.graphPredLabels || []);
const graphOtherPredLabels = computed(() => theOtherDs.graphPredLabels || []);
const graphColorScale = ds.graphColorScale || (() => "#888");

const localLinksArr = computed(() => db.srcLinksArr || []);
const localLinksDict = computed(() => db.srcLinksDict || {});
const mapLinkId2Index = computed<Record<Type_NodeId, number>>(
    () =>
        //分组（各graph）且过滤后（仅当前db有）的links，其id在db.srcLinksArr中的索引是什么
        localLinksArr.value.reduce(
            (acc, cur, curIndex) => ({
                ...acc,
                [cur.eid]: curIndex,
            }),
            {}
        ) //graphCoordsRet不需要过滤，我们只是从中取
);

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
const curSrcEntryId = ref<Type_NodesSelectionEntryId>(srcEntryIds[0]);
const srcNodesDict = computed(() => db.nodesSelections[curSrcEntryId.value]);
const localNodes = computed(
    () => db.graphCoordsRet.filter((d) => srcNodesDict.value[d.id]) || []
);
//NOTE 对于生成新dashboard，若选的nodes很任意，不构成完整graph的，我们不生成完整的graph。
const srcCompleteGraphsDict = computed(
    () => nodeMapGraph2GraphMapNodes(srcNodesDict.value) //完整graph
);
const localGraphArr = computed(() => {
    const ret = [];
    for (const graphId in srcCompleteGraphsDict.value) {
        const filteredLinks = ds.graphArr[+graphId].links //REVIEW
            .filter((linkId) => db.srcLinksDict[linkId]);
        const filteredNodes = ds.graphArr[+graphId].nodes.filter(
            (nodeId) => db.srcNodesDict[nodeId]
        );
        ret.push({ gid: graphId, filteredLinks, filteredNodes });
    }
    return ret;
});
const mapNodeId2Index = computed<Record<Type_NodeId, number>>(() =>
    //分组（各graph）且过滤后（仅当前db有）的nodes，其id在graphCoordsRet中的某索引是什么
    (db.graphCoordsRet as NodeCoord[]).reduce(
        (acc, cur, curI) => ({
            ...acc,
            [cur.id]: curI,
        }),
        {}
    )
);
const mapGroupedNodeId2Index = computed<Record<Type_NodeId, number>>(
    () =>
        //分组（各graph）且过滤后（仅当前db有）的nodes，其id在localGraphArr[graph].filteredNodes中的某索引是什么
        localGraphArr.value.reduce(
            (acc, cur) => ({
                ...acc,
                ...cur.filteredNodes.reduce(
                    (innerAcc, innerCur, innerCurI) => ({
                        ...innerAcc,
                        [innerCur]: innerCurI,
                    }),
                    {}
                ),
            }),
            {}
        ) //graphCoordsRet不需要过滤，我们只是从中取
);

const tgtEntryIds = myStore.getViewTargetNodesSelectionEntry(props.viewName);
const selectedGraphIdsByNodes = computed(() => {
    return nodeMapGraph2GraphMapNodes(db.nodesSelections[tgtEntryIds[0]]); //REVIEW why 0?
});
const renderEntry = computed(() => db.nodesSelections[tgtEntryIds[0]]);

const highlightedGraphIdsByNodes = computed(() => {
    return nodeMapGraph2GraphMapNodes(
        db.highlightedNodeIds as (typeof db.nodesSelections)[string]
    );
});

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
////////////////// SECTION scales & sizes
const mL = computed(() => view.bodyWidth * 0.01);
const mR = computed(() => view.bodyWidth * 0.01);
const mT = computed(() => mL.value); //scroll
const mB = computed(() => mR.value); //scroll
const contentWidth = computed(
    () => Math.max(view.bodyWidth - mL.value - mR.value, 0) //很有可能是负的
);
const gap = computed(() => Math.min(view.bodyHeight, view.bodyWidth) * 0.01);
const numGraphs = computed(() => localGraphArr.value.length);
const numColumns = computed(() => view.numColumns);
const numRows = computed(() => Math.ceil(numGraphs.value / view.numColumns));
const subWidth = computed(
    () =>
        (contentWidth.value - (numColumns.value - 1) * gap.value) /
        numColumns.value
);
const contentHeight = computed(
    () => (numRows.value - 1) * gap.value + numRows.value * view.subHeight //很有可能是负的
);
const chartHeight = computed(() => contentHeight.value + mT.value + mB.value);
watch(
    () => view.isAlignHeightAndWidth,
    (newV) => {
        if (newV) {
            view.subHeight = subWidth.value;
        }
    },
    { immediate: true }
);
const subInnerMarginTop = computed(() => view.subHeight * 0.05);
const subInnerMarginBottom = computed(() => view.subHeight * 0.05);
const subInnerMarginLeft = computed(() => subWidth.value * 0.05);
const subInnerMarginRight = computed(() => subWidth.value * 0.05);
const subInnerHeight = computed(() =>
    Math.max(
        view.subHeight - subInnerMarginTop.value - subInnerMarginBottom.value,
        0
    )
);
const subInnerWidth = computed(
    () => subWidth.value - subInnerMarginLeft.value - subInnerMarginRight.value
);
////////////////// !SECTION scales & sizes
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION points first calc
watch(
    [localGraphArr],
    ([newGraphArr]) => {
        view.groupedRescaledCoords = newGraphArr.map(
            ({ gid, filteredNodes, filteredLinks }) => ({
                gid: gid,
                filteredLinks: filteredLinks,

                filteredNodes: rescaleCoords<NodeCoord, never>(
                    filteredNodes.map(
                        (nodeId) =>
                            db.graphCoordsRet[mapNodeId2Index.value[nodeId]]
                    ),
                    [0, subInnerWidth.value],
                    [0, subInnerHeight.value],
                    (d) => d.x,
                    (d) => d.y,

                    db.name + "===" + props.viewName
                ),
            })
        );
    },
    { immediate: true, deep: true }
);
const flatNodes = computed(() =>
    view.groupedRescaledCoords.flatMap(({ filteredNodes }, i) =>
        filteredNodes.map((d) => ({
            ...d,
            x:
                mL.value +
                (gap.value + subWidth.value) * (i % numColumns.value) +
                subInnerMarginLeft.value +
                d.x,
            y:
                mT.value +
                Math.floor(i / numColumns.value) *
                    (gap.value + view.subHeight) +
                subInnerMarginTop.value +
                d.y,
        }))
    )
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
        view.groupedRescaledCoords = localGraphArr.value.map(
            ({ gid, filteredNodes, filteredLinks }) => ({
                gid: gid,
                filteredLinks: filteredLinks,

                filteredNodes: rescaleCoords<NodeCoord, never>(
                    filteredNodes.map(
                        (nodeId) =>
                            db.graphCoordsRet[mapNodeId2Index.value[nodeId]]
                    ),
                    [0, subInnerWidth.value],
                    [0, subInnerHeight.value],
                    (d) => d.x,
                    (d) => d.y,

                    db.name + "===" + props.viewName
                ),
            })
        );
    },
    db.name + "===" + props.viewName
);
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
            [view.bodyWidth, chartHeight.value],
        ],
        () => [
            [mL.value, mT.value],
            [contentWidth.value, contentHeight.value],
        ],
        flatNodes,
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
const transformRef = ref({ x: 0, y: 0, k: 1 });
// const { transformRef, enablePanFuncRef, disablePanFuncRef, resetZoomFuncRef } =
//     useD3Zoom(
//         svgRef,
//         undefined,
//         () => view.bodyWidth,
//         chartHeight,
//         (w, h) => [
//             [0, 0],
//             [w, h],
//         ],
//         undefined,
//         undefined,
//         !view.isBrushEnabled,
//         db.name + "===" + props.viewName,
//         true
//     );
// view.panEnableFunc = enablePanFuncRef;
// view.panDisableFunc = disablePanFuncRef;
// view.resetZoomFunc = resetZoomFuncRef;
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
