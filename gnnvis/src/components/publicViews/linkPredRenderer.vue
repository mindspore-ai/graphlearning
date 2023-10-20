<template>
    <div
        :style="{
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            height: view.bodyHeight + 'px',
        }"
    >
        <!-- v-if="isEmptyDict(db.nodesSelections['public'])" -->
        <PendingComp
            v-if="graphWorkerStatus === 'PENDING'"
            text="waiting for nodes selection"
        />
        <LoadingComp
            v-else-if="graphWorkerStatus === 'RUNNING'"
            text="calculating"
        />
        <ErrorComp
            v-else-if="graphWorkerStatus === 'ERROR' || error"
            :error="error"
        />
        <ErrorComp
            v-else-if="graphWorkerStatus === 'TIMEOUT_EXPIRED'"
            error="calc timeout!"
        />
        <template v-else-if="!isHiding">
            <svg id="legend" :width="view.bodyWidth" :height="legendSvgHeight">
                <g
                    :transform="`translate(${
                        view.bodyWidth * view.bodyMargins.left
                    } ${legendSvgHeight * view.bodyMargins.top})`"
                >
                    <g
                        v-for="(symbol, i) in legendSymbols"
                        :key="i"
                        :transform="`translate(${
                            i * (legendRectHeight + legendHorizontalMargin)
                        } ${0})`"
                    >
                        <g
                            :transform="`translate(${legendRectHeight / 2} ${
                                legendRectHeight / 2
                            })`"
                        >
                            <path
                                stroke="black"
                                :d="d3.symbol(symbol, legendRectHeight)()!"
                                fill="black"
                            ></path>
                        </g>
                        <text
                            :y="legendRectHeight + legendTextMargin"
                            :x="legendRectHeight / 2"
                            text-anchor="middle"
                            :font-size="legendTextHeight"
                            :dy="0.3 * legendTextHeight"
                        >
                            {{ legendSemantics[i] }}
                        </text>
                    </g>
                </g>
            </svg>
            <svg id="main" :width="view.bodyWidth" :height="mainHeight" v-zoom>
                <g
                    id="globalTransform"
                    :transform="`translate(${transformRef.x},${transformRef.y}) scale(${transformRef.k})`"
                >
                    <g id="links" stroke-linecap="round">
                        <template v-if="view.isShowGroundTruth">
                            <line
                                v-for="l in groundTruthLinks"
                                :key="l.eid"
                                :stroke-width="2"
                                stroke="#555"
                                :stroke-opacity="
                                    db.isHighlightCorrespondingNode &&
                                    !isEmptyDict(db.highlightedNodeIds)
                                        ? 0.1
                                        : 1
                                "
                                :x1="renderCoords[mapNodeId2Index[l.source]].x"
                                :y1="renderCoords[mapNodeId2Index[l.source]].y"
                                :x2="renderCoords[mapNodeId2Index[l.target]].x"
                                :y2="renderCoords[mapNodeId2Index[l.target]].y"
                            ></line>
                        </template>
                        <template v-if="view.isShowTrueAllow">
                            <template v-for="(l, i) in trueAllowEdges" :key="i">
                                <line
                                    v-if="l[0] != l[1]"
                                    :stroke-width="1"
                                    stroke="black"
                                    :x1="renderCoords[mapNodeId2Index[l[0]]].x"
                                    :y1="renderCoords[mapNodeId2Index[l[0]]].y"
                                    :x2="renderCoords[mapNodeId2Index[l[1]]].x"
                                    :y2="renderCoords[mapNodeId2Index[l[1]]].y"
                                ></line>
                                <circle
                                    v-else-if="view.isShowSelfLoop"
                                    :stroke-width="1"
                                    stroke="black"
                                    fill="none"
                                    :cx="
                                        getCenterOfSelfLoop(
                                            renderCoords[mapNodeId2Index[l[0]]]
                                                .x,
                                            renderCoords[mapNodeId2Index[l[0]]]
                                                .y,
                                            2 * view.nodeRadius
                                        )[0]
                                    "
                                    :cy="
                                        getCenterOfSelfLoop(
                                            renderCoords[mapNodeId2Index[l[0]]]
                                                .x,
                                            renderCoords[mapNodeId2Index[l[0]]]
                                                .y,
                                            2 * view.nodeRadius
                                        )[1]
                                    "
                                    :r="2 * view.nodeRadius"
                                ></circle>
                            </template>
                        </template>
                        <template v-if="view.isShowFalseAllow">
                            <template
                                v-for="(l, i) in falseAllowEdges"
                                :key="i"
                            >
                                <line
                                    v-if="l[0] != l[1]"
                                    :stroke-width="1"
                                    stroke="red"
                                    :x1="renderCoords[mapNodeId2Index[l[0]]].x"
                                    :y1="renderCoords[mapNodeId2Index[l[0]]].y"
                                    :x2="renderCoords[mapNodeId2Index[l[1]]].x"
                                    :y2="renderCoords[mapNodeId2Index[l[1]]].y"
                                ></line>
                                <circle
                                    v-else-if="view.isShowSelfLoop"
                                    :stroke-width="1"
                                    stroke="red"
                                    fill="none"
                                    :cx="
                                        getCenterOfSelfLoop(
                                            renderCoords[mapNodeId2Index[l[0]]]
                                                .x,
                                            renderCoords[mapNodeId2Index[l[0]]]
                                                .y,
                                            2 * view.nodeRadius
                                        )[0]
                                    "
                                    :cy="
                                        getCenterOfSelfLoop(
                                            renderCoords[mapNodeId2Index[l[0]]]
                                                .x,
                                            renderCoords[mapNodeId2Index[l[0]]]
                                                .y,
                                            2 * view.nodeRadius
                                        )[1]
                                    "
                                    :r="2 * view.nodeRadius"
                                ></circle>
                            </template>
                        </template>
                        <template v-if="view.isShowTrueUnseen">
                            <template v-for="(l, i) in unseenLinks" :key="i">
                                <line
                                    v-if="l.source != l.target"
                                    :stroke-width="dynamicStrokeWidth(l.top)"
                                    stroke="#409eff"
                                    stroke-dasharray="3 2"
                                    :x1="
                                        renderCoords[mapNodeId2Index[l.source]]
                                            .x
                                    "
                                    :y1="
                                        renderCoords[mapNodeId2Index[l.source]]
                                            .y
                                    "
                                    :x2="
                                        renderCoords[mapNodeId2Index[l.target]]
                                            .x
                                    "
                                    :y2="
                                        renderCoords[mapNodeId2Index[l.target]]
                                            .y
                                    "
                                ></line>
                                <circle
                                    v-else-if="view.isShowSelfLoop"
                                    :stroke-width="dynamicStrokeWidth(l.top)"
                                    stroke="#409eff"
                                    stroke-dasharray="3 2"
                                    :x1="
                                        renderCoords[mapNodeId2Index[l.source]]
                                            .x
                                    "
                                    fill="none"
                                    :cx="
                                        getCenterOfSelfLoop(
                                            renderCoords[
                                                mapNodeId2Index[l.source]
                                            ].x,
                                            renderCoords[
                                                mapNodeId2Index[l.source]
                                            ].y,
                                            2 * view.nodeRadius
                                        )[0]
                                    "
                                    :cy="
                                        getCenterOfSelfLoop(
                                            renderCoords[
                                                mapNodeId2Index[l.source]
                                            ].x,
                                            renderCoords[
                                                mapNodeId2Index[l.source]
                                            ].y,
                                            2 * view.nodeRadius
                                        )[1]
                                    "
                                    :r="2 * view.nodeRadius"
                                ></circle>
                            </template>
                        </template>
                    </g>
                    <g
                        id="points"
                        stroke="none"
                        stroke-opacity="1"
                        stroke-width="1"
                    >
                        <template v-for="n in renderCoords" :key="n.id">
                            <g :transform="`translate(${n.x} ${n.y})`">
                                <path
                                    class="outer"
                                    :d="getSymbolPathByNodeId(n.id)"
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
                                    :opacity="dynamicOpacity(n.id)"
                                    @mouseenter="
                                        db.highlightedNodeIds[n.id] = true
                                    "
                                    @mouseleave="db.highlightedNodeIds = {}"
                                >
                                    <title>
                                        {{
                                            `id: ${n.id}\ntrueLabel: ${
                                                labels[+n.id]
                                            }`
                                        }}
                                    </title>
                                </path>
                                <path
                                    v-if="isDbWiseComparativeDb(db)"
                                    class="LR"
                                    :d="getLRSymbolPathByNodeId(n.id)!"
                                    stroke-linejoin="bevel"
                                    :fill="colorScale(labels[+n.id] + '')"
                                    :stroke="
                                        renderEntry[n.id] ? 'black' : 'none'
                                    "
                                    :opacity="dynamicOpacity(n.id)"
                                    @mouseenter="
                                        db.highlightedNodeIds[n.id] = true
                                    "
                                    @mouseleave="db.highlightedNodeIds = {}"
                                >
                                    <title>
                                        {{
                                            `id: ${n.id}\ntrueLabel: ${
                                                labels[+n.id]
                                            }`
                                        }}
                                    </title>
                                </path>
                            </g>
                        </template>
                    </g>
                    <g class="gBrush" ref="brushRef"></g>
                </g>
            </svg>
        </template>
    </div>
</template>

<script setup lang="ts">
import { useMyStore } from "@/stores/store";
import * as d3 from "d3";
import type {
    CompDashboard,
    NodeView,
    SingleDashboard,
    Type_NodeId,
    NodeCoord,
    Node,
    LinkPredView,
} from "@/types/types";
import {
    computed,
    watch,
    ref,
    onMounted,
    onUnmounted,
    onBeforeUnmount,
    toRaw,
} from "vue";
import type { Ref } from "vue";
import { useResize } from "../plugin/useResize";
import {
    calcGraphCoords,
    calcNeighborDict,
    isEmptyDict,
    rescaleCoords,
} from "@/utils/graphUtils";
import PendingComp from "../state/Pending.vue";
import LoadingComp from "../state/Loading.vue";
import ErrorComp from "../state/Error.vue";
import { useWebWorkerFn } from "@/utils/myWebWorker";
import BitSet from "bitset";
import {
    circlePath,
    crossPath,
    getAreaBySymbolOuterRadius,
    leftHalfCirclePath,
    leftHalfCrossPath,
    rightHalfCirclePath,
    rightHalfCrossPath,
} from "@/utils/otherUtils";
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
    which: {
        type: Number,
        default: 0,
        required: true,
    },
});
////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION data prepare
const myStore = useMyStore();
const db =
    props.which == 0
        ? myStore.getSingleDashboardById(props.dbId)
        : myStore.getCompDashboardById(props.dbId);
const thisDs =
    props.which == 0
        ? myStore.getDatasetByName((db as SingleDashboard).refDatasetName)
        : props.which == 1
        ? myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[0])
        : myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[1]);
const theOtherDs =
    props.which == 0
        ? thisDs
        : props.which == 1
        ? myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[1])
        : myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[0]);
const view = myStore.getViewByName(db, props.viewName) as NodeView<
    NodeCoord & d3.SimulationNodeDatum
> &
    LinkPredView;
view.sourceCoords = []; //initial
db.nodesSelections["public2"] = {}; //REVIEW -  bad code!

const isHiding = ref(false); //NOTE - links比nodes算的快,所以要让svg先被屏蔽,防止link坐标出现空值
watch(
    [
        () => db.nodesSelections["public"],
        () => view.currentHops,
        () => view.isShowTrueUnseen,
    ],
    () => {
        isHiding.value = true;
    },
    { deep: true }
);

//已选点+ 当前hops（如当前hop为2则包括1，2）结点
const selNodesAndCurHopsNeighbor = computed(() => {
    const ret = {} as Record<
        Type_NodeId,
        { gid: string; parentDbIndex: number; hop: number }
    >;
    const selAndNeighbor = calcNeighborDict(
        db.nodesSelections["public"],
        view.currentHops,
        thisDs.neighborMasksByHop || theOtherDs.neighborMasksByHop!
    );
    for (const id in selAndNeighbor) {
        if (db.srcNodesDict[id]) {
            //NOTE 仅过滤当前db有的
            ret[id] = {
                ...selAndNeighbor[id],
                parentDbIndex: isDbWiseComparativeDb(db)
                    ? db.srcNodesDict[id].parentDbIndex
                    : 0,
                gid: (thisDs.globalNodesDict || theOtherDs!.globalNodesDict)[id]
                    .gid,
                //REVIEW - 直接用db.srcNodesDict是否保险？
            };
        }
    }

    return ret;
});

//已选点+ 当前hops（如当前hop为2则包括1，2）结点
// const selNodesAndCurHopsNeighbor = computed<
//     Record<Type_NodeId, { gid: string; parentDbIndex: number; hop: number }>
// >(() => {
//     const ret = {} as Record<
//         Type_NodeId,
//         { gid: string; parentDbIndex: number; hop: number }
//     >;
//     for (const id in selAndNeighborsDict.value) {
//         if (selAndNeighborsDict.value[id].hop < view.currentHops) {
//             ret[id] = selAndNeighborsDict.value[id];
//         }
//     }
//     return ret;
// });

const getLRSymbolPathByNodeId = (id: Type_NodeId) => {
    if (view.isShowTrueUnseen && unseenNodesInDb.value.has(id)) {
        switch (selNodesAndCurHopsNeighbor.value[id].parentDbIndex) {
            case 0:
                return leftHalfCrossPath(view.nodeRadius);
            case 1:
                return rightHalfCrossPath(view.nodeRadius);
            case 2:
                return crossPath(view.nodeRadius);
        }
    } else if (selNodesAndCurHopsNeighbor.value[id].hop === -1) {
        switch (selNodesAndCurHopsNeighbor.value[id].parentDbIndex) {
            case 0:
                return leftHalfCirclePath(view.nodeRadius);
            case 1:
                return rightHalfCirclePath(view.nodeRadius);
            case 2:
                return circlePath(view.nodeRadius);
        }
    } else {
        const { hop, parentDbIndex } = selNodesAndCurHopsNeighbor.value[id];
        return myStore.symbolPathByHopAndParentIndex[hop][parentDbIndex](
            view.nodeRadius
        );
    }
};
const getSymbolPathByNodeId = (id: Type_NodeId) => {
    if (view.isShowTrueUnseen && unseenNodesInDb.value.has(id)) {
        return crossPath(view.nodeRadius);
    } else if (selNodesAndCurHopsNeighbor.value[id].hop === -1) {
        return circlePath(view.nodeRadius);
    } else {
        const { hop } = selNodesAndCurHopsNeighbor.value[id];
        return myStore.symbolPathByHopAndParentIndex[hop][2](view.nodeRadius);
    }
};

const renderEntry = computed(() => db.nodesSelections["public2"]);
const mapNodeId2Index = computed<Record<Type_NodeId, number>>(() =>
    renderCoords.value.reduce(
        (acc, cur, curIndex) => ({
            ...acc,
            [cur.id]: curIndex,
        }),
        {}
    )
);

const groundTruthLinks = computed(() =>
    db.srcLinksArr.filter(
        (d) =>
            selNodesAndCurHopsNeighbor.value[d.target] &&
            selNodesAndCurHopsNeighbor.value[d.source]
    )
);
const trueAllowEdges = computed(() =>
    thisDs.trueAllowEdges.filter(
        (d) =>
            selNodesAndCurHopsNeighbor.value[d[0]] &&
            selNodesAndCurHopsNeighbor.value[d[1]]
    )
);
const falseAllowEdges = computed(() =>
    thisDs.falseAllowEdges.filter(
        (d) =>
            selNodesAndCurHopsNeighbor.value[d[0]] &&
            selNodesAndCurHopsNeighbor.value[d[1]]
    )
);

/**
 * 以selection 为key，过滤那些unseen。\
 * NOTE 仅考虑针对selection的推荐边，而不考虑selection的hopNeighbor的
 */
const unseenForSelected = computed<Record<Type_NodeId, Type_NodeId[]>>(() =>
    Object.keys(selNodesAndCurHopsNeighbor.value).reduce(
        (acc, cur) => ({ ...acc, [cur]: thisDs.trueUnseenEdgesSorted[cur] }),
        {}
    )
);

/**
 * 仅selection的unseen，且是当前view的topK（<=dataset.topK)长度下的，且是这些unseen必须是当前db有的
 */
const unseenNodesInDb = computed(
    () =>
        // d3.intersection(
        new Set(
            Object.values(unseenForSelected.value)
                .flatMap(
                    (d) => d.slice(0, view.numTrueUnseen).map((d) => d + "") //NOTE maybe number
                )
                .filter((id) => db.srcNodesDict[id])
        )
    // Object.keys(db.srcNodesDict)
    // )//d3.intersection
);

/**
 * unseen结点（限由selection选出的，限存在于此db的、限curTopK）与 sel+curHopsNeighbor 的并集。用于计算layout
 */
const unseenAndSelAndNeighbor = computed(() =>
    // d3.union(
    //     unseenNodesInDb.value,
    //     Object.keys(selNodesAndCurHopsNeighbor.value)
    // )
    {
        const ret = { ...selNodesAndCurHopsNeighbor.value };
        for (const nodeId of unseenNodesInDb.value) {
            if (db.srcNodesDict[nodeId]) ret[nodeId] = db.srcNodesDict[nodeId];
        }
        return ret;
    }
);

const unseenLinks = computed(() =>
    Object.keys(selNodesAndCurHopsNeighbor.value).flatMap((id) =>
        thisDs.trueUnseenEdgesSorted[id]
            .slice(0, view.numTrueUnseen)
            .map((unseenId, i) => ({
                eid: `unseen-${i}`,
                source: id,
                target: unseenId + "", //NOTE maybe number
                top: i + 1,
            })) //i是topK
            .filter((link) => unseenNodesInDb.value.has(link.target))
    )
);

//REVIEW :why we don't use srcEntry? "Prediction Space " is a little different.
const localNodesDict = computed(() =>
    view.isShowTrueUnseen
        ? unseenAndSelAndNeighbor.value
        : selNodesAndCurHopsNeighbor.value
);
const nodesForCalc = computed(() =>
    db.graphCoordsRet
        .filter((d) => localNodesDict.value[d.id])
        .map((d) => toRaw(d))
);
const linksForCalc = computed(() =>
    view.isShowTrueUnseen
        ? [
              ...groundTruthLinks.value.map((d) => toRaw(d)),
              ...unseenLinks.value.map((d) =>
                  toRaw(
                      d
                      //NOTE d3-force区分id是number还是string
                  )
              ),
          ]
        : groundTruthLinks.value.map((d) => toRaw(d))
);

const labels = computed(() => thisDs.trueLabels || theOtherDs.trueLabels || []);
////////////////// !SECTION data prepare
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION scale & sizes
const colorScale = thisDs.colorScale || theOtherDs.colorScale || (() => "#888");
const dynamicOpacity = computed(
    () => (i: string) =>
        !db.isHighlightCorrespondingNode || isEmptyDict(db.highlightedNodeIds)
            ? 1
            : db.highlightedNodeIds[i]
            ? 1
            : 0.1
);
const dynamicStrokeWidth = computed(
    () => (topK: number) =>
        d3.interpolate(
            1,
            Math.max(5, thisDs.trueUnseenTopK) //totalTopK比较少的时候也能尽可能粗一点
        )(topK / thisDs.trueUnseenTopK)
);
const legendSvgHeight = computed(() => 0.1 * view.bodyHeight);
const legendHorizontalMargin = computed(() => 0.02 * view.bodyWidth);
const legendRectHeight = computed(() =>
    Math.min(
        0.6 * legendSvgHeight.value,
        (view.bodyWidth * (1 - view.bodyMargins.right - view.bodyMargins.left) -
            (1 + 2 + view.currentHops) * legendHorizontalMargin.value) /
            (2 + view.currentHops)
    )
);
const legendTextMargin = computed(() => 0.1 * legendSvgHeight.value);
const legendTextHeight = computed(() => 0.2 * legendSvgHeight.value);
const legendSymbols = computed(() => [
    view.symbolSelection,
    view.symbolUnseen,
    ...myStore.symbolHops.slice(0, view.currentHops),
]);
const legendSemantics = computed(() => [
    "selection",
    "unseen",
    ...Array.from({ length: view.currentHops }, (_, i) => `hop-${i + 1}`),
]);

const mainHeight = computed(() => view.bodyHeight - legendSvgHeight.value);
const getCenterOfSelfLoop = (
    x: number,
    y: number,
    r: number
): [number, number] => {
    //自环边：计算环的圆心
    //svg正中心和(x,y)连线，来决定环的圆心
    const x0 =
        (view.bodyMargins.left * view.bodyWidth +
            view.bodyWidth * (1 - view.bodyMargins.right)) /
        2;
    const y0 =
        (view.bodyMargins.top * mainHeight.value +
            mainHeight.value * (1 - view.bodyMargins.bottom)) /
        2;
    const ratio = r / Math.sqrt((x - x0) ** 2 + (y - y0) ** 2);
    return [x - ratio * (x - x0), y - ratio * (y - y0)];
};
////////////////// !SECTION scale & sizes
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION calc graph
const error = ref<any>();
const {
    workerFn: graphWorkerFn,
    workerStatus: graphWorkerStatus,
    workerTerminate: graphWorkerTerminate,
} = useWebWorkerFn(calcGraphCoords, {
    timeout: 20_000,
    dependencies: [
        "http://localhost:5173/workers/d3js.org_d3.v7.js",
        // "https://d3js.org/d3.v7.js", //for debug
        // "https://d3js.org/d3.v7.min.js"
    ],
});
const calcPromiseKit: {
    p: Promise<void> | undefined;
    rejector: (reason?: any) => void;
} = {
    p: undefined,
    rejector: () => {},
};
let throttleTimer: ReturnType<typeof setTimeout> = -1;
watch(
    [nodesForCalc, linksForCalc],
    ([newNodes, newLinks]) => {
        clearTimeout(throttleTimer);
        throttleTimer = setTimeout(() => {
            // console.log("in link pred view, in calc args", newNodes, newLinks);
            error.value = undefined;
            calcPromiseKit.rejector("in link pred view, new calc encountered");
            graphWorkerTerminate();

            calcPromiseKit.p = new Promise<void>((resolve, reject) => {
                calcPromiseKit.rejector = reject;
                graphWorkerFn(
                    toRaw(newNodes),
                    toRaw(newLinks),
                    undefined, //use default iter
                    view.nodeRadius
                )
                    .then((ret) => {
                        graphWorkerTerminate();
                        // if (Math.random() < 0.6) {
                        console.log("in link pred view, calc got ret ", ret);
                        view.sourceCoords = ret;
                        resolve();
                        // } else throw new Error("god threw a coin");
                    })
                    .catch((e) => {
                        console.log(e);
                        error.value = new Error(
                            Object.hasOwn(e, "message") ? e.message : e
                        );
                        reject(e);
                    })
                    .finally(() => {
                        isHiding.value = false;
                    });
            });
        }, 300);
    },
    { deep: true, flush: "post" }
);
watch(
    () => view.sourceCoords,
    (newV) => {
        // console.log( "in db", db.name, "in linkPred, view.sourceCoords changed to", newV,);
        view.rescaledCoords = rescaleCoords<
            NodeCoord & d3.SimulationNodeDatum,
            never
        >(
            newV || [],
            [
                view.bodyMargins.left * view.bodyWidth,
                view.bodyWidth * (1 - view.bodyMargins.right),
            ],
            [
                view.bodyMargins.top * mainHeight.value,
                mainHeight.value * (1 - view.bodyMargins.bottom),
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
const renderCoords = computed(() => view.rescaledCoords);
////////////////// !SECTION calc graph
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION resize
//NOTE resize必须在brush和zoom之前，而brush和zoom谁前后无所谓
const { widthScaleRatio, heightScaleRatio } = useResize(
    () => view.resizeEndSignal,
    () => db.whenToRescaleMode === "onResizeEnd",
    () => view.bodyWidth,
    mainHeight,
    () => {
        if (graphWorkerStatus.value === "SUCCESS" && !error.value) {
            console.log(
                "in",
                props.dbId + props.viewName,
                "rescale Caller called!"
            );
            view.rescaledCoords = rescaleCoords<
                NodeCoord & d3.SimulationNodeDatum,
                never
            >(
                view.sourceCoords,
                [
                    view.bodyMargins.left * view.bodyWidth,
                    view.bodyWidth * (1 - view.bodyMargins.right),
                ],
                [
                    view.bodyMargins.top * mainHeight.value,
                    mainHeight.value * (1 - view.bodyMargins.bottom),
                ],
                (d: NodeCoord & d3.SimulationNodeDatum) => d.x,
                (d: NodeCoord & d3.SimulationNodeDatum) => d.y,
                db.name + "===" + props.viewName
            ) as (NodeCoord & Node & d3.SimulationNodeDatum)[];
        }
    },
    db.name + "===" + props.viewName
);
////////////////// !SECTION resize
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION zoom
const transformRef = ref(d3.zoomIdentity);

const zoom = d3
    .zoom<SVGElement, unknown>()
    .extent([
        [0, 0],
        [view.bodyWidth, mainHeight.value],
    ])
    .translateExtent([
        [0, 0],
        [view.bodyWidth, mainHeight.value],
    ])
    .scaleExtent([1 / 2, 16])
    .on("zoom", function (event: d3.D3ZoomEvent<SVGElement, void>) {
        // if (event.sourceEvent) {
        console.log("in predLink, on zoom!");
        transformRef.value = event.transform;
        // }
    });
const vZoom = {
    mounted(el: SVGElement) {
        console.log("in linkPred vZoom, mounted!", d3.select(el).node());
        d3.select(el).call(zoom, d3.zoomIdentity);

        view.resetZoomFunc = () => {
            d3.select(el).transition().duration(750).call(
                //我们仅需第一个即可，后面rect都是根据 transformRef 改的

                zoom.transform, //这个函数会调用onZoom
                d3.zoomIdentity
                // d3
                //     .zoomTransform(d3.zoomIdentity)
                //     .invertX(heatStripWidth.value / 2)
            );
        };
        view.panDisableFunc = () => {
            zoom.filter((event) => event.type === "wheel");
        };
        view.panEnableFunc = () => {
            zoom.filter(
                (event) =>
                    (!event.ctrlKey || event.type === "wheel") && !event.button //default
            );
        };
    },
    unmounted(el: SVGElement) {
        console.log("in linkPred vZoom, unMounted!", d3.select(el).node());
        // zoom.on("zoom", null);//不需要

        transformRef.value = d3.zoomIdentity;

        view.resetZoomFunc = () => {};
        view.panDisableFunc = () => {};
        view.panEnableFunc = () => {};
    },
};
watch(
    [() => view.bodyWidth, mainHeight],
    ([newW, newH]) => {
        zoom.extent([
            [0, 0],
            [newW, newH],
        ]).translateExtent([
            [0, 0],
            [newW, newH],
        ]);
    },
    { immediate: true }
);
////////////////// !SECTION zoom
////////////////////////////////////////////////////////////////////////////////
</script>

<style scoped></style>
