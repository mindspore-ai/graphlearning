<template>
    <div
        :style="{
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            height: view.bodyHeight + 'px',
        }"
    >
        <PendingComp
            v-if="
                !isDbWiseComparativeDb(db) &&
                isEmptyDict(db.nodesSelections['public'])
            "
            text="waiting for nodes selection"
        />
        <LoadingComp
            v-else-if="
                graphWorkerStatus === 'RUNNING' ||
                distWorkerStatus === 'RUNNING'
            "
            :text="
                graphWorkerStatus === 'RUNNING' &&
                distWorkerStatus === 'RUNNING'
                    ? 'calc layout & density...'
                    : graphWorkerStatus === 'RUNNING'
                    ? 'calc layout...'
                    : 'calc density...'
            "
        />
        <ErrorComp
            v-else-if="
                graphWorkerStatus === 'ERROR' ||
                graphWorkerStatus === 'TIMEOUT_EXPIRED' ||
                distWorkerStatus === 'ERROR' ||
                distWorkerStatus === 'TIMEOUT_EXPIRED' ||
                calcError
            "
            :error="calcError"
        />
        <svg
            v-else
            ref="svgRef"
            :width="view.bodyWidth"
            :height="view.bodyHeight"
            version="1.1"
            xmlns="http://www.w3.org/2000/svg"
            baseProfile="full"
        >
            <g
                class="links"
                v-if="view.isShowLinks"
                stroke="#999"
                :stroke-opacity="view.linkOpacity"
                stroke-linecap="round"
            >
                <line
                    v-for="(l, i) in localLinks"
                    :key="i"
                    :stroke-width="1"
                    :stroke-opacity="
                        db.isHighlightCorrespondingNode &&
                        !isEmptyDict(db.highlightedNodeIds)
                            ? 0.1
                            : view.linkOpacity
                    "
                    :x1="view.rescaledCoords[mapNodeId2Index[l.source]].x"
                    :y1="view.rescaledCoords[mapNodeId2Index[l.source]].y"
                    :x2="view.rescaledCoords[mapNodeId2Index[l.target]].x"
                    :y2="view.rescaledCoords[mapNodeId2Index[l.target]].y"
                ></line>
            </g>
            <g v-if="view.isShowHopSymbols">
                <template v-for="n in view.rescaledCoords" :key="n.id">
                    <g :transform="`translate(${n.x} ${n.y})`">
                        <path
                            class="outer"
                            :d="getSymbolPathByNodeId(n.id)!"
                            :fill="
                                isDbWiseComparativeDb(db)
                                    ? 'white'
                                    : colorScale(
                                          distData[distMapNodeId2Index[n.id]]
                                              .dist
                                      )
                            "
                            :stroke="
                                colorScale(
                                    distData[distMapNodeId2Index[n.id]].dist
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
                                    `id: ${n.id}\nlatent-diff: ${
                                        distData[distMapNodeId2Index[n.id]].dist
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
                                colorScale(
                                    distData[distMapNodeId2Index[n.id]].dist
                                )
                            "
                            :opacity="dynamicOpacity(n.id)"
                            :stroke="
                                colorScale(
                                    distData[distMapNodeId2Index[n.id]].dist
                                )
                            "
                            :stroke-width="1"
                            @mouseenter="
                                db.highlightedNodeIds[n.id] =
                                    db.srcNodesDict[n.id]
                            "
                            @mouseleave="db.highlightedNodeIds = {}"
                        >
                            <title>
                                {{
                                    `id: ${n.id}\nlatent-diff: ${
                                        distData[distMapNodeId2Index[n.id]].dist
                                    }`
                                }}
                            </title>
                        </path>
                    </g>
                </template>
            </g>
            <g v-else>
                <circle
                    v-for="n in view.rescaledCoords"
                    :key="n.id"
                    :r="view.nodeRadius"
                    :fill="colorScale(distData[distMapNodeId2Index[n.id]].dist)"
                    :cx="n.x"
                    :cy="n.y"
                    :stroke="'none'"
                    :opacity="1"
                    @mouseenter="
                        db.highlightedNodeIds[n.id] = db.srcNodesDict[n.id]
                    "
                    @mouseleave="db.highlightedNodeIds = {}"
                >
                    <title>
                        {{ `id: ${n.id}\n` }}
                    </title>
                </circle>
            </g>
        </svg>
    </div>
</template>

<script setup lang="ts">
import {
    onMounted,
    onUnmounted,
    watch,
    ref,
    computed,
    toRaw,
    type ComputedRef,
} from "vue";
import type {
    AggregatedView,
    LinkableView,
    Type_NodeId,
    CompDashboard,
    NodeCoord,
    SingleDashboard,
    Type_NodesSelectionEntryId,
    NodeView,
    Node,
    Type_GraphId,
} from "../../types/types";

import { useMyStore } from "../../stores/store";
import { useD3Brush } from "../plugin/useD3Brush";
import { useD3Zoom } from "../plugin/useD3Zoom";
import { useResize } from "../plugin/useResize";

import * as d3 from "d3";
import {
    calcGraphCoords,
    calcNeighborDict,
    calcVectorDist,
    isEmptyDict,
    rescaleCoords,
} from "../../utils/graphUtils";
import {
    circlePath,
    getAreaBySymbolOuterRadius,
    leftHalfCirclePath,
    rightHalfCirclePath,
} from "@/utils/otherUtils";
import LoadingComp from "../state/Loading.vue";
import PendingComp from "../state/Pending.vue";
import { useWebWorkerFn } from "@/utils/myWebWorker";
import type { Link } from "../../types/types";
import ErrorComp from "../state/Error.vue";
import BitSet from "bitset";
import { isDbWiseComparativeDb } from "@/types/typeFuncs";

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
        default: 1,
        required: true,
    },
});

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION data prepare
const svgRef = ref<SVGElement | null>(null);
const myStore = useMyStore();

const db = myStore.getTypeReducedDashboardById(props.dbId)!;
const thisDs =
    props.which == 0
        ? myStore.getDatasetByName((db as SingleDashboard).refDatasetName)
        : props.which == 1
        ? myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[0])
        : myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[1]);
const theOtherDs =
    props.which == 0
        ? undefined
        : props.which == 1
        ? myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[1])
        : myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[0]);
const view = myStore.getViewByName(db, props.viewName) as NodeView<NodeCoord> &
    LinkableView;
////////////////// !SECTION data prepare
////////////////////////////////////////////////////////////////////////////////

const srcEntryIds = myStore.getViewSourceNodesSelectionEntry(props.viewName);
const curSrcEntryId = ref<Type_NodesSelectionEntryId>(srcEntryIds[0]);
const srcNodesDict = computed(() => db.nodesSelections[curSrcEntryId.value]);

const localNodesAndNeighborDict = computed(() => {
    if (isDbWiseComparativeDb(db)) {
        return db.srcNodesDict;
    } else {
        const selAndNeighbor = calcNeighborDict(
            srcNodesDict.value,
            thisDs.hops || theOtherDs?.hops!,
            thisDs.neighborMasksByHop || theOtherDs?.neighborMasksByHop!
        );

        const ret: Record<
            Type_NodeId,
            { gid: string; parentDbIndex: number; hop: number }
        > = {};
        for (const id in selAndNeighbor) {
            if (db.srcNodesDict[id]) {
                //NOTE 仅过滤当前db有的
                ret[id] = {
                    ...selAndNeighbor[id],
                    parentDbIndex: 0,
                    gid: (thisDs.globalNodesDict ||
                        theOtherDs!.globalNodesDict)[id].gid,
                };
            }
        }
        return ret;
    }
});
const localLinks = computed(() =>
    isDbWiseComparativeDb(db)
        ? db.srcLinksArr
        : (thisDs.links! || theOtherDs!.links!).filter(
              (d) =>
                  localNodesAndNeighborDict.value[d.source] &&
                  localNodesAndNeighborDict.value[d.target]
          ) || []
);

const calcError = ref();
const {
    workerFn: graphWorkerFn,
    workerStatus: graphWorkerStatus,
    workerTerminate: graphWorkerTerminate,
} = useWebWorkerFn(
    calcGraphCoords<
        Node,
        Link & d3.SimulationLinkDatum<Node & d3.SimulationNodeDatum>
    >,
    {
        timeout: 20_000,
        dependencies: ["http://localhost:5173/workers/d3js.org_d3.v7.js"],
    }
);
const {
    workerFn: distWorkerFn,
    workerStatus: distWorkerStatus,
    workerTerminate: distWorkerTerminate,
} = useWebWorkerFn(calcVectorDist, {
    timeout: 20_000,
    dependencies: [
        "http://localhost:5173/workers/d3js.org_d3.v7.js",
        "http://localhost:5173/workers/distance.js",
    ],
});
const distData = ref<{ id: Type_NodeId; dist: number }[]>([]);
const distDataExtent = computed(() =>
    d3.extent<number>(distData.value.map((d) => d.dist))
);
const distMapNodeId2Index = ref<Record<Type_NodeId, number>>({});
const colorScale = (d: number) =>
    view.isRelativeColorScale //NOTE 如果把三元表达式写在函数外，则丢失响应性
        ? d3.interpolate(
              d3.interpolateGreys(0.1), //NOTE - 0.1是因为最小值0看起来太白了
              d3.interpolateGreys(1)
          )(
              (d - (distDataExtent.value[0] || 0)) /
                  ((distDataExtent.value[1] || 1) -
                      (distDataExtent.value[0] || 0))
          )
        : d3.interpolate(d3.interpolateGreys(0.1), d3.interpolateGreys(1))(d);
watch(
    localNodesAndNeighborDict,
    async (newNodesDict) => {
        if (graphWorkerStatus.value === "RUNNING") {
            graphWorkerTerminate();
        }
        if (distWorkerStatus.value === "RUNNING") {
            distWorkerTerminate();
        }
        calcError.value = null;
        try {
            if (!isDbWiseComparativeDb(db)) {
                const calcNodes =
                    toRaw(
                        thisDs.nodes || theOtherDs!.nodes
                        // db.graphCoordsRet// NOTE maybe incomplete
                    )?.filter((d) => newNodesDict[d.id]) || [];
                const calcLinks =
                    toRaw(
                        thisDs.links || theOtherDs!.links
                        // db.srcLinksArr
                    )?.filter(
                        (d) => newNodesDict[d.source] && newNodesDict[d.target]
                    ) || [];

                // console.log(
                //     "in topo latent density view, READY to calc layout, Nodes&Links",
                //     calcNodes,
                //     calcLinks,
                //     "\nREADY to calc dist, toRaw(db.srcNodesDict) is",
                //     toRaw(db.srcNodesDict),
                //     "\ntoRaw(newNodesDict)",
                //     toRaw(db.srcNodesDict),
                //     "\nembAll",
                //     toRaw(thisDs.embNode || theOtherDs?.embNode)
                // );
                view.sourceCoords = await graphWorkerFn(calcNodes, calcLinks);

                distData.value = await distWorkerFn(
                    Object.keys(newNodesDict).filter((d) => newNodesDict[d]),
                    Object.keys(db.srcNodesDict).filter(
                        (d) => db.srcNodesDict[d]
                    ),
                    toRaw(thisDs.embNode),
                    true
                );
                distMapNodeId2Index.value = distData.value.reduce(
                    (acc, cur, curI) => ({
                        ...acc,
                        [cur.id]: curI,
                    }),
                    {}
                );
            } else {
                view.sourceCoords = db.graphCoordsRet;
                const nodesArr = Object.keys(toRaw(newNodesDict));
                const nodesForCalc0 = nodesArr.filter(
                    (d) =>
                        newNodesDict[d] && newNodesDict[d].parentDbIndex === 0
                );
                const nodesForCalc1 = nodesArr.filter(
                    (d) =>
                        newNodesDict[d] && newNodesDict[d].parentDbIndex === 1
                );
                const nodesExemptCalc = nodesArr.filter(
                    (d) =>
                        newNodesDict[d] && newNodesDict[d].parentDbIndex === 2
                );

                // console.log(
                //     "in topo latent density view, dbWiseComp MODE, ready to calc dist",
                //     "\nnodesArr is",
                //     nodesArr,
                //     "\nnodesForCalc0 is",
                //     nodesForCalc0,
                //     "\nnodesForCalc1 is",
                //     nodesForCalc1,
                //     "\nnodesExemptCalc is",
                //     nodesExemptCalc
                // );
                const dist01 = await distWorkerFn(
                    nodesForCalc0,
                    nodesForCalc1,
                    toRaw(thisDs.embNode),
                    true
                );
                const dist10 = await distWorkerFn(
                    nodesForCalc1,
                    nodesForCalc0,
                    toRaw(thisDs.embNode),
                    true
                );
                const distExempt = nodesExemptCalc.map((id) => ({
                    id,
                    dist: 0,
                }));
                distData.value = [...dist01, ...dist10, ...distExempt];
                distMapNodeId2Index.value = distData.value.reduce(
                    (acc, cur, curI) => ({
                        ...acc,
                        [cur.id]: curI,
                    }),
                    {}
                );
            }

            console.log(
                "in topo latent density view, calcCoords, got ret",
                view.sourceCoords,
                "\ncalcDist, got ret",
                distData.value
            );
        } catch (e) {
            console.error(
                "in topo - latent density view, calc layout but got error",
                e
                // typeof e,
                // e.message,
                // typeof e.message
            );
            calcError.value =
                e && typeof e === "object" && Object.hasOwn(e, "message")
                    ? e.message
                    : e;
        } finally {
            graphWorkerTerminate();
        }
    },
    { immediate: isDbWiseComparativeDb(db), deep: true }
);
watch(
    () => view.sourceCoords,
    (newV) => {
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
                view.bodyMargins.top * view.bodyHeight,
                view.bodyHeight * (1 - view.bodyMargins.bottom),
            ],
            (d) => d.x,
            (d) => d.y,
            db.name + "===" + props.viewName
        ) as (NodeCoord & d3.SimulationNodeDatum)[];
    },
    { immediate: isDbWiseComparativeDb(db), deep: true }
);
const mapNodeId2Index = computed<Record<Type_NodeId, number>>(() =>
    view.rescaledCoords.reduce(
        (acc, cur, curIndex) => ({
            ...acc,
            [cur.id]: curIndex,
        }),
        {}
    )
);

const getSymbolPathByNodeId = (id: Type_NodeId) => {
    if (localNodesAndNeighborDict.value[id].hop === -1) {
        return circlePath(view.nodeRadius);
    } else {
        const { hop: hopIndex } = localNodesAndNeighborDict.value[id];
        return myStore.symbolPathByHopAndParentIndex[hopIndex][2](
            view.nodeRadius
        );
    }
};
const getLRSymbolPathByNodeId = (id: Type_NodeId) => {
    if (localNodesAndNeighborDict.value[id].hop === -1) {
        switch (localNodesAndNeighborDict.value[id].parentDbIndex) {
            case 0:
                return leftHalfCirclePath(view.nodeRadius);
            case 1:
                return rightHalfCirclePath(view.nodeRadius);
            case 2:
                return circlePath(view.nodeRadius);
        }
    } else {
        const { hop: hopIndex, parentDbIndex } =
            localNodesAndNeighborDict.value[id];
        return myStore.symbolPathByHopAndParentIndex[hopIndex][parentDbIndex](
            view.nodeRadius
        );
    }
};

const dynamicOpacity = computed(
    () => (i: string) =>
        !db.isHighlightCorrespondingNode || isEmptyDict(db.highlightedNodeIds)
            ? 1
            : db.highlightedNodeIds[i]
            ? 1
            : 0.1
);
</script>

<style scoped></style>
