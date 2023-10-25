<!-- known bugs: reselecting srcNodes won't trigger recalc in aggr mode -->
<template>
    <svg
        ref="svgRef"
        :width="view.bodyWidth"
        :height="view.bodyHeight"
        :viewBox="`${-view.bodyWidth / 2 + mL} 
            ${-view.bodyHeight / 2 + mT} ${view.bodyWidth} ${view.bodyHeight}`"
        version="1.1"
        xmlns="http://www.w3.org/2000/svg"
        baseProfile="full"
    >
        <g id="margin" :transform="`translate(${mL},${mT})`">
            <g
                id="svgContent"
                :transform="`translate(${transformRef.x},${transformRef.y}) scale(${transformRef.k})`"
            >
                <!--i从1开始，所以减1。从外向内画。外部颜色更淡，毕竟hop越远影响力越小-->
                <circle
                    class="background"
                    v-for="i in view.hops + 1"
                    :key="i"
                    :cx="0"
                    :cy="0"
                    :r="rHopsStart(view.hops + 1 - (i - 1))"
                    :fill="getBgColor(i - 1)"
                ></circle>
                <line
                    :x1="0"
                    :y1="-R"
                    :x2="0"
                    :y2="R"
                    stroke="currentColor"
                ></line>
                <line
                    :x1="-R"
                    :y1="0"
                    :x2="R"
                    :y2="0"
                    stroke="currentColor"
                ></line>

                <g
                    id="hexbinMesh"
                    v-if="view.hexbin && view.isShowMesh"
                    stroke="#ccc"
                    fill="none"
                    stroke-width="0.5"
                >
                    <path :d="view.hexbin.mesh()"></path>
                </g>

                <template v-if="view.isShowLinks">
                    <g
                        v-if="view.isShowAggregation"
                        id="aggregatedLinks"
                        stroke="#999"
                        :stroke-opacity="view.linkOpacity || 0"
                        stroke-linecap="round"
                    >
                        <line
                            v-for="l in view.aggregatedLinks"
                            :key="l.aeid"
                            :stroke-width="d3.scaleLinear().domain(
                            d3.extent(
                                view.aggregatedLinks?.map((d) => d.baseLinks.length) || []
                            ) as [number, number]
                        )
                        .range([2, 8])(l.baseLinks.length)
                        "
                            :x1="view.aggregatedCoords[l.source].x"
                            :y1="view.aggregatedCoords[l.source].y"
                            :x2="view.aggregatedCoords[l.target].x"
                            :y2="view.aggregatedCoords[l.target].y"
                        ></line>
                    </g>
                    <g
                        v-else
                        id="links"
                        stroke="#999"
                        :stroke-opacity="view.linkOpacity || 0"
                        stroke-linecap="round"
                    >
                        <line
                            v-for="(l, i) in view.localLinks"
                            :key="i"
                            :stroke-width="1"
                            :x1="
                                view.cartesianCoords[mapNodeId2Index[l.source]]
                                    .x
                            "
                            :y1="
                                view.cartesianCoords[mapNodeId2Index[l.source]]
                                    .y
                            "
                            :x2="
                                view.cartesianCoords[mapNodeId2Index[l.target]]
                                    .x
                            "
                            :y2="
                                view.cartesianCoords[mapNodeId2Index[l.target]]
                                    .y
                            "
                        ></line>
                    </g>
                </template>

                <g v-if="view.isShowAggregation" id="aggregatedPoints">
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
                                        db.labelType === "pred"
                                            ? "pred"
                                            : "true"
                                    }Label: ${arc.data[0]}\ncount: ${
                                        arc.data[1]
                                    }`
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
                                            Number(
                                                renderAggregatedSelEntry[d.id]
                                            )
                                        )
                                    )
                                "
                            ></path>
                        </g>
                    </g>
                </g>
                <g v-else id="points" stroke="none" :stroke-width="1.5">
                    <circle
                        v-for="n in view.cartesianCoords"
                        :key="n.id"
                        :r="2"
                        :stroke="
                            db.nodesSelections[tgtEntryIds[0]][n.id]
                                ? 'black'
                                : 'none'
                        "
                        :fill="colorScale(curLabels[+n.id] + '')"
                        :cx="n.x"
                        :cy="n.y"
                        :opacity="dynamicOpacity(n.id)"
                        @mouseenter="db.highlightedNodeIds[n.id] = true"
                        @mouseleave="db.highlightedNodeIds = {}"
                    >
                        <title>
                            {{
                                `id: ${n.id}\n
                                ${
                                    db.labelType === "pred" ? "pred" : "true"
                                }Label: ${curLabels[+n.id]}\n
                                embDiff: ${
                                    view.polarData
                                        .find((d) => d.id === n.id)
                                        ?.embDiff.toFixed(3) || "not found"
                                }\n
                                topoDist: ${
                                    view.polarData
                                        .find((d) => d.id == n.id)
                                        ?.topoDist.toFixed(3) || "not found"
                                }`
                            }}
                        </title>
                    </circle>
                </g>
                <g class="gBrush" ref="brushRef"></g>
            </g>
        </g>

        <!-- 放在#margin之外，zoom绑定#margin，这样就能避免双击switch会触发zoom，放在margin下方，是为了防止遮挡-->
        <g
            id="switch"
            :transform="`translate(${mL + (-view.bodyWidth * 3) / 8},${
                mT + (-view.bodyHeight * 3) / 8
            })`"
            cursor="pointer"
            @click.stop="handleSwitchClick"
        >
            <rect
                width="110"
                height="50"
                fill="none"
                stroke="black"
                stroke-width="2"
            ></rect>
            <text stroke="currentColor" font-size="1em" dy="1.8em" dx="0.4em">
                click to switch
            </text>
        </g>
    </svg>
</template>

<script setup lang="ts">
import {
    onMounted,
    onUnmounted,
    watch,
    ref,
    computed,
    type Ref,
    type ComputedRef,
    watchEffect,
} from "vue";
import { useMyStore } from "@/stores/store";
import {
    type Type_NodesSelectionEntryId,
    type PolarDatum,
    type PolarView,
    type Type_NodeId,
    type Type_ClusterId,
    type PolarCoord,
    type Link,
    type Type_LinkId,
    type NodeCoord,
} from "@/types/types";
import { useD3Brush } from "../plugin/useD3Brush";
import { useD3Zoom } from "../plugin/useD3Zoom";
import { useResize } from "../plugin/useResize";
import { isEmptyDict, rescalePolarCoords } from "../../utils/graphUtils";
import * as d3 from "d3";
import { watchDebounced } from "@vueuse/core";

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
});

const svgRef = ref<SVGElement | null>(null);
const myStore = useMyStore();
onMounted(() => {
    console.warn("in", props.viewName, "onMounted!");
});
onUnmounted(() => {
    console.warn("in", props.viewName, "onUnmounted!");
});

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION data prepare
const db = myStore.getCompDashboardById(props.dbId)!;
const view = myStore.getViewByName(props.dbId, props.viewName) as PolarView;
const ds1 = myStore.getDatasetByName(db.refDatasetsNames[0])!;
const ds2 = myStore.getDatasetByName(db.refDatasetsNames[1])!;

const nodeMapLink = ds1.nodeMapLink || ds2.nodeMapLink || [];

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
console.log(
    "in viewName",
    props.viewName,
    "polar renderer, got view.polarData",
    view.polarData
);
const localNodesRecord = computed<Record<Type_NodeId, boolean>>(() =>
    view.polarData.reduce(
        (acc, cur) => ({
            ...acc,
            [cur.id]: true,
        }),
        {}
    )
);

const mapNodeId2Index = computed<Record<Type_NodeId, number>>(() =>
    view.polarData.reduce(
        (acc, cur, curIndex) => ({
            ...acc,
            [cur.id]: curIndex,
        }),
        {}
    )
);

////////////////// !SECTION data prepare
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION nodeSelectionEntry related
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
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION sizes & scales
const mL = computed(() => view.bodyWidth * view.bodyMargins.left);
const mR = computed(() => view.bodyWidth * view.bodyMargins.right);
const mT = computed(() => view.bodyHeight * view.bodyMargins.top);
const mB = computed(() => view.bodyHeight * view.bodyMargins.bottom);
const rectWidth = computed(
    () => Math.max(view.bodyWidth - mL.value - mR.value, 0) //很有可能是负的
);
const rectHeight = computed(
    () => Math.max(view.bodyHeight - mT.value - mB.value, 0) //很有可能是负的
);
const R = computed(() => Math.min(rectHeight.value, rectWidth.value) / 2);
watchEffect(() => {
    view.R = R.value;
});
const bgColor = d3.interpolate("#f1f1f1", "#979797"); //由浅到深
const getBgColor = (hop: number) =>
    bgColor(hop / (view.hops > 4 ? view.hops : 4)); //NOTE hops较小时也按照更细腻的阶数插值
const rHopsStart = computed(
    () => (i: number) => (i / (view.hops + 1)) * R.value
); //i从0开始，每个hop对应r的开始点，例如3个hop，0:0R, 1:1R/3, 2:2R/3.

const colorScale = ds1.colorScale || ds2.colorScale || (() => "#888");
const dynamicOpacity = computed(
    () => (i: string) =>
        !db.isHighlightCorrespondingNode || isEmptyDict(db.highlightedNodeIds)
            ? 1
            : db.highlightedNodeIds[i]
            ? 1
            : 0.1
);
////////////////// !SECTION sizes & scales
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION calc points
// const radialScale = computed(() => d3.scaleRadial().domain().range()); //NOTE calc in rescale
// const angleScale = computed(() => d3.scaleLinear().domain().range()); //NOTE calc in rescale
const mapPolarToX = (radius: number, angle: number) =>
    radius * Math.cos(-angle); //NOTE: 上下翻转，其实是负的角度，而不是负的极径
// NOTE not an omnipotent math func, but merely applicable for this view
const mapPolarToY = (radius: number, angle: number) =>
    radius * Math.sin(-angle);
const getMetaphorRadius = ref((d: PolarDatum) => d.topoDist); //NOTE 定义成ref是为switch做准备
const getMetaphorAngle = ref((d: PolarDatum) => d.embDiff);
const getRadiusRangeByHop = computed(
    //NOTE 数据中hop从0开始，即hop=0为一阶邻居。但是vue的`<template>`的`v-for`以1开始
    () => (hop: number) =>
        [rHopsStart.value(hop + 1), rHopsStart.value(hop + 2)] as [
            number,
            number
        ]
);
// watch(
//     rHopsStart, //这个watcher为了把这个函数提取到公共。因为header组件中也有rescale，要用到函数
//     // 不想提取也可以，把header组件中的rescale逻辑提取到这个组件
//     (newV) => {
//         getRadiusRangeByHop.value = (hop: number) =>
//             [newV(hop), newV(hop + 1)] as [number, number];
//     },
//     { immediate: true, deep: true }
// );
watch(
    [() => view.polarData, getMetaphorRadius, getMetaphorAngle],
    ([newD, newRFn, newAFn]) => {
        view.localLinks = db.srcLinksArr.filter(
            (d) =>
                localNodesRecord.value[d.source] &&
                localNodesRecord.value[d.target]
        );

        view.sourceCoords =
            newD?.map((d) => ({
                // ...d,//REVIEW 要保留原始信息吗
                id: d.id,
                radius: newRFn(d),
                angle: newAFn(d),
                hop: d.hop,
            })) || [];
    },
    { immediate: true, deep: true }
);
const handleSwitchClick = () => {
    console.log("in polar, switch clicked");
    let tmp = getMetaphorRadius.value;
    getMetaphorRadius.value = getMetaphorAngle.value;
    getMetaphorAngle.value = tmp;
    // event.stopPropagation();
};
watch(
    [() => view.sourceCoords, () => view.isShowAggregation],
    ([newSrc, newIsAgg]) => {
        if (!newIsAgg) {
            //REVIEW 将isShowAggregation变化为false之后极就rescale的逻辑从header提取到了这里
            view.rescaledPolarCoords = rescalePolarCoords<PolarCoord>(
                newSrc,
                undefined, //default
                undefined, //default
                getRadiusRangeByHop.value,
                () => [0, Math.PI],
                (d) => d.hop!
            );
        }
    },
    { immediate: true, deep: true }
);
watch(
    () => view.rescaledPolarCoords,
    (newV) => {
        view.cartesianCoords = newV.map((d) => ({
            ...d,
            x: mapPolarToX(d.radius, d.angle),
            y: mapPolarToY(d.radius, d.angle),
        }));
    },
    { immediate: true, deep: true }
);
////////////////// !SECTION calc points
////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION resize
const { widthScaleRatio, heightScaleRatio } = useResize(
    () => view.resizeEndSignal,
    () => db.whenToRescaleMode === "onResizeEnd",
    () => view.bodyWidth,
    () => view.bodyHeight,
    () => {
        //在resize时立刻调用，不管当前是否在isAggregation
        view.rescaledPolarCoords = rescalePolarCoords<PolarCoord>(
            view.sourceCoords,
            undefined,
            undefined,
            getRadiusRangeByHop.value,
            () => [0, Math.PI],
            (d) => d.hop!
        );
    },
    db.name + "===" + props.viewName
);
watch(
    [() => view.cartesianCoords, () => db.labelType],
    ([newV, newLabelType]) => {
        if (view.isShowAggregation) {
            myStore.calcAggregatedViewData(
                view,
                () => newV,
                () => [
                    [
                        view.bodyMargins.left * view.bodyWidth - view.R,
                        view.bodyMargins.top * view.bodyHeight - view.R,
                    ],
                    [
                        view.R - view.bodyMargins.right * view.bodyWidth,
                        0 - view.bodyMargins.bottom * view.bodyHeight,
                    ],
                ],
                ds1.taskType === "node-classification"
                    ? newLabelType === "true"
                        ? trueLabels
                        : predLabels.value
                    : trueLabels,
                true,
                view.localLinks
            );
        }
    },
    { immediate: true, deep: true }
);
////////////////// !SECTION resize
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION zoom
const { transformRef, enablePanFuncRef, disablePanFuncRef, resetZoomFuncRef } =
    useD3Zoom(
        svgRef,
        "#margin", //让出switch元素
        () => view.bodyWidth,
        () => view.bodyHeight,
        (w, h) => [
            [-w / 2, -h / 2],
            [w / 2, h / 2],
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

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION brush
const brushRef = ref<SVGGElement | null>(null);
const { enableBrushFuncRef, disableBrushFuncRef, hideBrushRectFuncRef } =
    useD3Brush<NodeCoord>( //NOTE 泛型应当写NodeCoords，不然下面的根据条件传入points类型会编译不通过。
        brushRef,
        () => [
            [-view.bodyWidth / 2, -view.bodyHeight / 2],
            [view.bodyWidth / 2, view.bodyHeight / 2],
        ],
        () => [
            [mL.value - view.bodyWidth / 2, mT.value - view.bodyHeight / 2],
            [view.bodyWidth / 2 - mR.value, view.bodyHeight / 2 - mB.value],
        ],
        () =>
            view.isShowAggregation
                ? view.aggregatedCoords
                : view.cartesianCoords,
        () => {
            tgtEntryIds.forEach((entryId) => {
                db.nodesSelections[entryId] = {};
            });
        },
        (id: Type_NodeId | Type_ClusterId) => {
            db.fromViewName = props.viewName;
            if (view.isShowAggregation) {
                const { pointIds } = view.clusters.find((d) => d.id === id) || {
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
                    db.nodesSelections[entryId][id] = db.srcNodesDict[id];
                });
            }
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
</script>

<style scoped></style>
