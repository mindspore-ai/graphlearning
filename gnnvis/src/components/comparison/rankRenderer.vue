<template>
    <svg
        ref="svgRef"
        :width="view.bodyWidth"
        :height="view.bodyHeight"
        version="1.1"
        xmlns="http://www.w3.org/2000/svg"
        baseProfile="full"
    >
        <!-- :view-box="`${left} ${top} ${view.bodyWidth} ${view.bodyHeight}`" -->
        <!-- <g class="margin" :transform="`translate(${(.left, .top)})`"></g> -->
        <!-- REVIEW 要不要移动margin,即便移动了left和top，还有bottom和right等着你呢 -->
        <!-- <g class="debug" :transform="`translate(100 100)`"> <text>{{ widthScaleRatio }}, {{ heightScaleRatio }}</text> </g> -->
        <g class="legend"></g>
        <g class="tipText"></g>
        <g
            class="xAxisLabel"
            :transform="`translate(${view.bodyWidth * 0.75}, 
                ${view.bodyHeight * 0.97})`"
        >
            <text :font-size="Math.trunc(mB * 0.4)">
                rank in latent space 1 ({{
                    ds1.name.split("-").at(-1) || ds1.name
                }})
            </text>
        </g>
        <g
            class="yAxisLabel"
            :transform="`translate(${view.bodyWidth * 0.05}, ${mT * 0.8})`"
        >
            <text :font-size="Math.trunc(mT * 0.9)">
                rank in latent space 2 ({{
                    ds2.name.split("-").at(-1) || ds2.name
                }})
            </text>
        </g>

        <g
            class="xAxisTicks"
            :transform="`translate(0, ${view.bodyHeight - mB})`"
            font-size="1em"
            text-anchor="middle"
            fill="none"
        >
            <g
                v-for="(d, i) in xTicks"
                :key="i"
                :transform="`translate(${zoomedXScale(d)},0)`"
            >
                <line stroke="currentColor" :y2="-6"></line>
                <text fill="currentColor" :y="15" dy="0em">{{ d }}</text>
            </g>
        </g>

        <g
            class="yAxisTicks"
            :transform="`translate(${mL}, 0)`"
            font-size="1em"
            text-anchor="end"
            fill="none"
        >
            <g
                v-for="(d, i) in yTicks"
                :key="i"
                :transform="`translate(0,${zoomedYScale(d)})`"
            >
                <line stroke="currentColor" :x2="-6"></line>
                <text fill="currentColor" :x="-7" dy="0.3em">
                    {{ d }}
                </text>
            </g>
        </g>

        <g
            class="globalTransform"
            stroke="none"
            :stroke-width="1.5"
            :transform="`translate(${transformRef.x},${transformRef.y}) scale(${transformRef.k})`"
        >
            <g
                class="hexbinMesh"
                v-if="view.hexbin && view.isShowMesh"
                stroke="#ccc"
                fill="none"
                stroke-width="0.5"
            >
                <path :d="view.hexbin.mesh()"></path>
            </g>

            <rect
                class="frame"
                stroke="currentColor"
                :stroke-width="1"
                fill="none"
                :x="mL"
                :y="mT"
                :width="rectWidth"
                :height="rectHeight"
            ></rect>
            <line
                ref="line1Ref"
                class="line1"
                cursor="move"
                stroke="currentColor"
                :stroke-width="5"
                stroke-linecap="butt"
                :x1="mL"
                :y1="y1Line1"
                :x2="x2Line1"
                :y2="mT"
            >
                <title>drag in pan mode</title>
            </line>
            <line
                ref="line2Ref"
                class="line2"
                cursor="move"
                stroke="currentColor"
                :stroke-width="5"
                stroke-linecap="butt"
                :x1="x1Line2"
                :y1="Math.max(view.bodyHeight - mB, 0)"
                :x2="Math.max(view.bodyWidth - mR, 0)"
                :y2="y2Line2"
            >
                <title>drag in pan mode</title>
            </line>
            <g v-if="view.isShowAggregation" class="aggregatedPoints">
                <g
                    v-for="d in view.aggregatedCoords"
                    :key="d.id"
                    class="pie"
                    :transform="`translate(${d.x}, ${d.y})`"
                    stroke="black"
                    stroke-width="0"
                    @mouseenter="
                        view.clusters[+d.id].pointIds.forEach(
                            (id) => (db.highlightedNodeIds[id] = true)
                        )
                    "
                    @mouseleave="db.highlightedNodeIds = {}"
                >
                    <g v-for="(arc, i) in d.arcs" :key="i">
                        <title>
                            {{
                                `${
                                    db.labelType === "pred" ? "pred" : "true"
                                }Label: ${arc.data[0]}\ncount: ${arc.data[1]}`
                            }}
                        </title>

                        <path
                            :fill="colorScale(arc.data[0])"
                            :d="arcFuncFunc(d)(arc)!"
                            :stroke-width="0"
                            stroke="none"
                        ></path>
                        <path
                            :d="
                                outerArcFunc(d)()?.slice(
                                    0,
                                    outerArcFunc(d)()?.indexOf('L')
                                ) || ''
                            "
                            :stroke-width="2"
                            fill="none"
                            :stroke="
                                d3.interpolate(
                                    '#888',
                                    '#000'
                                )(
                                    Math.sqrt(
                                        Number(renderAggregatedSelEntry[d.id])
                                    )
                                )
                            "
                        ></path>
                    </g>
                </g>
            </g>

            <g v-else class="points" stroke-width="1.5">
                <circle
                    v-for="d in view.rescaledCoords"
                    :key="d.id"
                    :stroke="renderSingleSelEntry[d.id] ? 'black' : 'none'"
                    :fill="colorScale(curLabels[+d.id] + '')"
                    r="2"
                    :cx="d.x"
                    :cy="d.y"
                    :opacity="dynamicOpacity(d.id)"
                    @mouseenter="db.highlightedNodeIds[d.id] = true"
                    @mouseleave="db.highlightedNodeIds = {}"
                >
                    <title>
                        {{
                            `id: ${d.id}\n${
                                db.labelType === "pred" ? "pred" : "true"
                            }Label: ${curLabels[+d.id]}`
                        }}
                    </title>
                </circle>
            </g>
            <g class="gBrush" ref="brushRef"></g>
        </g>
    </svg>
</template>

<script setup lang="ts">
//NOTE 为了更规范使用生命周期，将svg作为一个组件抽离出来非常有必要。

import {
    onMounted,
    watch,
    ref,
    computed,
    type Ref,
    type ComputedRef,
    onUnmounted,
} from "vue";
import { useMyStore } from "@/stores/store";
import type { Type_NodeId, Type_NodesSelectionEntryId } from "@/types/types";
import { useD3Brush } from "../plugin/useD3Brush";
import { useD3Zoom } from "../plugin/useD3Zoom";
import { useResize } from "../plugin/useResize";
import { isEmptyDict, rescaleCoords } from "../../utils/graphUtils";
import * as d3 from "d3";
import type { RankView, NodeCoord, Type_ClusterId } from "@/types/types";

const props = defineProps({
    dbId: {
        type: String,
        default: "",
        required: true,
    },
    viewName: {
        type: String,
        default: "Graph",
    },
});

const svgRef = ref<SVGElement | null>(null);
const myStore = useMyStore();

//////////////////////////////////////////////////////////////////
////////////////// SECTION data prepare
const db = myStore.getCompDashboardById(props.dbId)!;
const view = myStore.getViewByName(props.dbId, props.viewName) as RankView;
const ds1 = myStore.getDatasetByName(db.refDatasetsNames[0])!;
const ds2 = myStore.getDatasetByName(db.refDatasetsNames[1])!;

const trueLabels = ds1?.trueLabels || ds2?.trueLabels || [];
const predLabels = computed(() =>
    view.isUsingSecondDbPredLabel ? ds2.predLabels || [] : ds1.predLabels || []
);
const curLabels = computed(() =>
    ds1.taskType === "node-classification"
        ? db.labelType === "true"
            ? trueLabels
            : predLabels.value
        : trueLabels
);
////////////////// !SECTION data prepare
//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
////////////////// SECTION nodeSelectionEntry related
//因为rank view的数据源是在外部计算好的、然后通过props或者store传进来的。而不是在db.nodesSelection中选择的。
// const srcEntryIds = myStore.getViewSourceNodesSelectionEntry(props.viewName);
// const curSrcEntryId = ref<Type_NodesSelectionEntryId>(srcEntryIds[0]);
// const srcNodesDict = computed(() => db.nodesSelections[curSrcEntryId.value]);

const tgtEntryIds = myStore.getViewTargetNodesSelectionEntry(props.viewName);

const renderSingleSelEntry = computed(() => db.nodesSelections[tgtEntryIds[0]]); //REVIEW why 0?
const renderAggregatedSelEntry: ComputedRef<Record<Type_NodeId, number>> =
    computed(() =>
        view.clusters.reduce(
            (acc, cur) => ({
                ...acc,
                [cur.id]:
                    cur.pointIds.reduce((inAcc, inCur) => {
                        if (db.nodesSelections[tgtEntryIds[0]][inCur]) inAcc++;
                        return inAcc; //求交集长度
                    }, 0) / cur.pointIds.length, //求交集占有率
            }),
            {} //最终得到{clusterId: 占有率}
        )
    );
////////////////// !SECTION nodeSelectionEntry related
//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
////////////////// SECTION sizes & scales
const mL = computed(() => view.bodyWidth * view.bodyMargins.left);
const mR = computed(() => view.bodyWidth * view.bodyMargins.right);
const mT = computed(() => view.bodyHeight * view.bodyMargins.top);
const mB = computed(() => view.bodyHeight * view.bodyMargins.bottom);
const rectHeight = computed(
    () => Math.max(view.bodyHeight - mT.value - mB.value, 0) //可能是负的
);
const rectWidth = computed(
    () => Math.max(view.bodyWidth - mL.value - mR.value, 0) //可能是负的
);
const colorScale = ds1.colorScale || ds2.colorScale || (() => "#888");
const dynamicOpacity = computed(
    () => (i: string) =>
        !db.isHighlightCorrespondingNode || isEmptyDict(db.highlightedNodeIds)
            ? 1
            : db.highlightedNodeIds[i]
            ? 1
            : 0.1
);
const xScale = computed(
    () =>
        d3
            .scaleLinear()
            .range([mL.value, view.bodyWidth - mR.value])
            .domain(
                d3.extent(view.rankData.map((d) => d.r1) || []) as [
                    number,
                    number
                ]
            )
    // .nice()
);
const yScale = computed(
    () =>
        d3
            .scaleLinear()
            .range([view.bodyHeight - mB.value, mT.value])
            .domain(
                d3.extent((view.rankData || []).map((d) => d.r2)) as [
                    number,
                    number
                ]
            )
    // .nice()
);
const zoomedYScale = computed(() =>
    transformRef.value.rescaleY(yScale.value).interpolate(d3.interpolateRound)
);
const zoomedXScale = computed(() =>
    transformRef.value.rescaleX(xScale.value).interpolate(d3.interpolateRound)
);
const ticksNum = 12;
const xTicks = computed(() => zoomedXScale.value.ticks(ticksNum));
const yTicks = computed(() =>
    zoomedYScale.value.ticks((ticksNum * view.bodyHeight) / view.bodyWidth)
);
////////////////// !SECTION sizes & scales
//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
////////////////// SECTION coords first calc
watch(
    () => view.rankData,
    (newV) => {
        view.sourceCoords = newV.map((d) => ({ id: d.id, x: d.r1, y: d.r2 }));
    },
    {
        deep: true,
        immediate: true,
    }
);
watch(
    () => view.sourceCoords,
    (newV) => {
        view.rescaledCoords = rescaleCoords(
            newV,
            [mL.value, view.bodyWidth - mR.value],
            [view.bodyHeight - mB.value, mT.value],
            (d: NodeCoord) => d.x,
            (d: NodeCoord) => d.y,
            db.name + "===" + props.viewName
        );
    },
    {
        immediate: true,
        deep: true,
    }
);
////////////////// !SECTION coords first calc
//////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION aggregated points
const arcFuncFunc = <
    T extends { id: string; x: number; y: number; mag: number },
    U extends d3.PieArcDatum<any>
>(
    d: T
) => d3.arc<U>().innerRadius(0).outerRadius(view.clusterRadiusScale(d.mag));
const outerArcFunc = <
    T extends {
        id: string;
        x: number;
        y: number;
        mag: number;
    }
>(
    d: T
) =>
    d3
        .arc<void>()
        .innerRadius(0)
        .outerRadius(view.clusterRadiusScale(d.mag) * 1.1)
        .startAngle(0)
        .endAngle(Number(renderAggregatedSelEntry.value[d.id] * 2 * Math.PI));

//NOTE 应当是在switch isAggregate 之后，立刻resize一次。放在了header中
////////////////// !SECTION aggregated points
////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
////////////////// SECTION resize
//NOTE resize必须在brush和zoom之前，而brush和zoom谁前后无所谓
const { widthScaleRatio, heightScaleRatio } = useResize(
    () => view.resizeEndSignal,
    () => db.whenToRescaleMode === "onResizeEnd",
    () => view.bodyWidth,
    () => view.bodyHeight,
    () => {
        //在resize之后立刻rescale，不管isShowAggregation如何。
        //让aggregatedCoords自动依赖rescaledCoords计算
        view.rescaledCoords = rescaleCoords<NodeCoord, never>(
            view.sourceCoords as NodeCoord[],
            [mL.value, view.bodyWidth - mR.value],
            [view.bodyHeight - mB.value, mT.value],
            (d: NodeCoord) => d.x,
            (d: NodeCoord) => d.y,
            db.name + "===" + props.viewName
        );
        // }
    },
    db.name + "===" + props.viewName
);
watch(
    [() => view.rescaledCoords, () => db.labelType],
    ([newV, newLabelType]) => {
        if (view.isShowAggregation) {
            myStore.calcAggregatedViewData(
                view,
                () => newV,
                () => [
                    [
                        view.bodyWidth * view.bodyMargins.left,
                        view.bodyHeight * (1 - view.bodyMargins.bottom),
                    ],
                    [
                        view.bodyWidth * (1 - view.bodyMargins.right),
                        view.bodyHeight * view.bodyMargins.top,
                    ],
                ],
                ds1.taskType === "node-classification"
                    ? newLabelType === "true"
                        ? trueLabels
                        : predLabels.value
                    : trueLabels,
                false,
                []
            );
        }
    }
);
////////////////// !SECTION resize
//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
////////////////// SECTION line
//NOTE y - top = -k * (x - left) + b
const k = computed(() =>
    view.bodyHeight && view.bodyWidth //边界检查
        ? -(mT.value - (view.bodyHeight - mB.value)) /
          (view.bodyWidth - mL.value - mR.value)
        : 1
); //NOTE 注意这个负号

const mouseXLine1 = ref(mL.value);
const mouseYLine1 = ref(mL.value + rectHeight.value / 2);
//line1 is the left line, [x1, y1] is the left bottom point
const y1Line1 = computed(
    () => k.value * (mouseXLine1.value - mL.value) + mouseYLine1.value
);
const x2Line1 = computed(
    () => (mouseYLine1.value - mT.value) / k.value + mouseXLine1.value
);
const dragged1 = (e: PointerEvent) => {
    console.log("in rank, dragged1!");
    if (
        e.y <= view.bodyHeight - mB.value - k.value * (e.x - mL.value) &&
        e.y >= mT.value - k.value * (e.x - mL.value)
    ) {
        mouseXLine1.value = e.x;
        mouseYLine1.value = e.y;
    } else if (e.y > view.bodyHeight - mB.value - k.value * (e.x - mL.value)) {
        mouseXLine1.value = mL.value; //左下角
        mouseYLine1.value = view.bodyHeight - mB.value;
    } else {
        mouseXLine1.value = mL.value; //左上角
        mouseYLine1.value = mT.value;
    }
};
// const dragended1 = (e) => {
//     mouseXLine1.value = mL.value; //当结束时，将mouse的值存成line1最左边的点，这样resize就不会bug
//     mouseYLine1.value = y1Line1.value;
// };
const drag1 = d3.drag<SVGLineElement, void>().on("drag", dragged1);
// .on("end", dragended1);

const mouseXLine2 = ref(mL.value + rectWidth.value / 2);
const mouseYLine2 = ref(view.bodyHeight - mB.value);
//line 2 is the right line, [x1, y1] is the left point
const x1Line2 = computed(
    () =>
        (mouseYLine2.value - (view.bodyHeight - mB.value)) / k.value +
        mouseXLine2.value
);
const y2Line2 = computed(
    () =>
        k.value * (mouseXLine2.value - (view.bodyWidth - mR.value)) +
        mouseYLine2.value
);

const dragged2 = (e: PointerEvent) => {
    console.log("in rank dragged2!!!");
    if (
        e.y >= view.bodyHeight - mB.value - k.value * (e.x - mL.value) &&
        e.y <=
            view.bodyHeight -
                mB.value +
                k.value * (view.bodyWidth - mR.value - e.x)
    ) {
        mouseXLine2.value = e.x;
        mouseYLine2.value = e.y;
    } else if (e.y < view.bodyHeight - mB.value - k.value * (e.x - mL.value)) {
        //左下角
        mouseXLine2.value = mL.value;
        mouseYLine2.value = view.bodyHeight - mB.value;
    } else {
        //右下角

        mouseXLine2.value = view.bodyWidth - mR.value;
        mouseYLine2.value = view.bodyHeight - mB.value;
    }
};
const drag2 = d3.drag<SVGLineElement, void>().on("drag", dragged2);

const isBrushable = computed(() => (x: number, y: number) => {
    //y-top = -k(x-left) + b 两条线之外的可以brush
    return (
        y <= y1Line1.value - k.value * (x - mL.value) ||
        y >= y2Line2.value - k.value * (x - (view.bodyWidth - mR.value))
    );
});
watch([widthScaleRatio, heightScaleRatio], ([newWR, newHR]) => {
    if (newWR && newWR != Infinity) {
        mouseXLine1.value = newWR * mL.value;
        mouseXLine2.value = newWR * x1Line2.value;
    }
    if (newHR && newHR != Infinity) {
        mouseYLine1.value = newHR * y1Line1.value;
        mouseYLine2.value = newHR * view.bodyHeight - mB.value;
    }
});

onMounted(() => {
    console.warn("In rank view, mounted!");
    d3.select<SVGElement, void>(svgRef.value!)
        .select<SVGLineElement>("#line1")
        .call(drag1);
    d3.select<SVGElement, void>(svgRef.value!)
        .select<SVGLineElement>("#line2")
        .call(drag2);
});
onUnmounted(() => {
    console.warn("In rank view, unmounted!");
});
////////////////// !SECTION line
//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
////////////////// SECTION brush
const brushRef = ref<SVGGElement | null>(null);
const {
    enableBrushFuncRef,
    disableBrushFuncRef,
    hideBrushRectFuncRef,
    // selectedNodesDict,
} = useD3Brush(
    brushRef,
    () => [
        [0, 0],
        [view.bodyWidth, view.bodyHeight],
    ],
    () => [
        [mL.value, mT.value],
        [view.bodyWidth - mR.value, view.bodyHeight - mB.value],
    ],
    () =>
        view.isShowAggregation ? view.aggregatedCoords : view.rescaledCoords,

    () => {
        tgtEntryIds.forEach((entryId) => {
            db.nodesSelections[entryId] = {};
        });
        // localTargetEntry.value = {};
        // aggregatedLocalTargetEntry.value = {};
    },
    (id: Type_NodeId | Type_ClusterId) => {
        db.fromViewName = props.viewName;
        if (view.isShowAggregation) {
            const { pointIds } = view.clusters.find(
                (d) => d.id === (id as Type_ClusterId)
            ) || {
                pointIds: [],
            };
            pointIds.forEach((pointId) => {
                tgtEntryIds.forEach((entryId) => {
                    db.nodesSelections[entryId][pointId] =
                        db.srcNodesDict[pointId];
                });
            });
        } else {
            tgtEntryIds.forEach((entryId) => {
                db.nodesSelections[entryId][id as Type_NodeId] =
                    db.srcNodesDict[id];
            });
        }
        // if (view.isShowAggregation)
        //     aggregatedLocalTargetEntry.value[id as Type_ClusterId] = true;
        // else localTargetEntry.value[id as Type_NodeId] = true;
    },
    isBrushable,
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
//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
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
//////////////////////////////////////////////////////////////////
</script>

<style scoped></style>
