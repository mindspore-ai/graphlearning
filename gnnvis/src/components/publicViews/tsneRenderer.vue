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
                        v-for="(l, i) in links"
                        :key="i"
                        :stroke-width="1"
                        :x1="view.rescaledCoords[mapNodeId2Index[l.source]].x"
                        :y1="view.rescaledCoords[mapNodeId2Index[l.source]].y"
                        :x2="view.rescaledCoords[mapNodeId2Index[l.target]].x"
                        :y2="view.rescaledCoords[mapNodeId2Index[l.target]].y"
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
                        view.clusters[+d.id].pointIds.forEach((id) => {
                            db.highlightedNodeIds[id] = srcNodesDict[id];
                        })
                    "
                    @mouseleave="db.highlightedNodeIds = {}"
                >
                    <g v-for="(arc, i) in d.arcs" :key="i">
                        <title>
                            {{
                                (db.labelType === "true" ? "true" : "pred") +
                                `Label: ${arc.data[0]}\ncount: ${arc.data[1]}`
                            }}
                        </title>
                        <path
                            :fill="colorScale(arc.data[0])"
                            :d="arcFuncFunc(d)(arc) || ''"
                            :stroke-width="0"
                            stroke="none"
                        ></path>
                        <!-- <path
                            :d="outerArcs[i]"
                            stroke="black"
                            stroke-width="2"
                            fill="none"
                        ></path> -->
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

            <template v-else>
                <g v-if="view.isShowHopSymbols">
                    <template v-for="n in view.rescaledCoords" :key="n.id">
                        <g :transform="`translate(${n.x} ${n.y})`">
                            <path
                                class="outer"
                                :d="getSymbolPathByNodeId(n.id)!"
                                :fill="
                                    isDbWiseComparativeDb(db)
                                        ? 'white'
                                        : colorScale(curLabels[+n.id] + '')
                                "
                                :stroke="
                                    renderSingleSelEntry[n.id]
                                        ? 'black'
                                        : colorScale(curLabels[+n.id] + '')
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
                                        `id: ${n.id}\n` +
                                        (db.labelType === "true"
                                            ? "true"
                                            : "pred") +
                                        `Label: ${curLabels[+n.id]}`
                                    }}
                                </title>
                            </path>
                            <path
                                v-if="isDbWiseComparativeDb(db)"
                                class="LR"
                                :d="getLRSymbolPathByNodeId(n.id)!"
                                stroke-linejoin="bevel"
                                :fill="colorScale(curLabels[+n.id] + '')"
                                :stroke="
                                    renderSingleSelEntry[n.id]
                                        ? 'black'
                                        : colorScale(curLabels[+n.id] + '')
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
                                        `id: ${n.id}\n` +
                                        (db.labelType === "true"
                                            ? "true"
                                            : "pred") +
                                        `Label: ${curLabels[+n.id]}`
                                    }}
                                </title>
                            </path>
                        </g>
                    </template>
                </g>
                <g v-else>
                    <circle
                        v-for="d in view.rescaledCoords"
                        :key="d.id"
                        :stroke="renderSingleSelEntry[d.id] ? 'black' : 'none'"
                        stroke-width="1.5"
                        :fill="colorScale(curLabels[+d.id] + '')"
                        r="2"
                        :cx="d.x"
                        :cy="d.y"
                        :opacity="dynamicOpacity(d.id)"
                        @mouseenter="
                            db.highlightedNodeIds[d.id] = srcNodesDict[d.id]
                        "
                        @mouseleave="db.highlightedNodeIds = {}"
                    >
                        <title>
                            {{
                                `id: ${d.id}\n` +
                                (db.labelType === "true" ? "true" : "pred") +
                                `Label: ${curLabels[+d.id]}`
                            }}
                        </title>
                    </circle>
                </g>
            </template>
            <g class="gBrush" ref="brushRef"></g>
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
    type ComputedRef,
} from "vue";
import { isEmptyDict, rescaleCoords } from "../../utils/graphUtils";
import type {
    AggregatedView,
    LinkableView,
    Type_ClusterId,
    Type_NodeId,
} from "../../types/types";
import { useMyStore } from "../../stores/store";
import * as d3 from "d3";
import type {
    CompDashboard,
    NodeCoord,
    SingleDashboard,
    TsneCoord,
    Type_NodesSelectionEntryId,
} from "@/types/types";
import { useD3Brush } from "../plugin/useD3Brush";
import { useD3Zoom } from "../plugin/useD3Zoom";
import { useResize } from "../plugin/useResize";
import {
    circlePath,
    getAreaBySymbolOuterRadius,
    leftHalfCirclePath,
    rightHalfCirclePath,
} from "@/utils/otherUtils";
import { isDbWiseComparativeDb } from "@/types/typeFuncs";

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
        ? thisDs
        : props.which == 1
        ? myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[1])
        : myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[0]);
const view: AggregatedView & LinkableView = myStore.getViewByName(
    db,
    props.viewName
) as AggregatedView & LinkableView;
console.log("in db", db, " in tsne View, view is", view);

const taskType = thisDs.taskType;
const predLabels = thisDs.predLabels || [];
const trueLabels = thisDs.trueLabels || theOtherDs?.trueLabels || [];
const curLabels = computed(() =>
    taskType === "node-classification"
        ? db.labelType === "true"
            ? trueLabels
            : predLabels
        : trueLabels
);
const links = computed(() => db.srcLinksArr || []);
const colorScale =
    thisDs.colorScale || theOtherDs?.colorScale || (() => "#888");

const dynamicOpacity = computed(
    () => (i: string) =>
        !db.isHighlightCorrespondingNode || isEmptyDict(db.highlightedNodeIds)
            ? 1
            : db.highlightedNodeIds[i]
            ? 1
            : 0.1
);
const tsneRet = computed(() =>
    props.which == 1
        ? (db as CompDashboard).tsneRet1
        : props.which == 2
        ? (db as CompDashboard).tsneRet2
        : (db as SingleDashboard).tsneRet || []
);

const mapNodeId2Index = computed<Record<Type_NodeId, number>>(() =>
    tsneRet.value.reduce(
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
const srcEntryIds = myStore.getViewSourceNodesSelectionEntry(props.viewName);
const tgtEntryIds = myStore.getViewTargetNodesSelectionEntry(props.viewName);

// const localTargetEntry: Ref<Record<Type_ClusterId, boolean>> = ref({});
// NOTE  local target sel, then watch + modify all db target entries
//      but this will incur problems: all views change db entries by their localTargetSel
//      so the db entries will be inconsistent
//      thus, we directly change db target entries and remove local entry
//      for rendering, we get data from db target entries instead of local entry

const renderSingleSelEntry = computed(() => db.nodesSelections[tgtEntryIds[0]]); //REVIEW why 0?
const renderAggregatedSelEntry: ComputedRef<Record<Type_NodeId, number>> =
    computed(() =>
        view.clusters
            // .filter(
            //     (clu) =>
            //         clu.pointIds.every((id) => localTargetEntry.value[id]) // we only filter 100%?
            // )
            .reduce(
                (acc, cur) => ({
                    ...acc,
                    [cur.id]:
                        cur.pointIds.reduce((inAcc, inCur) => {
                            if (db.nodesSelections[tgtEntryIds[0]][inCur])
                                inAcc++;
                            return inAcc; //求交集长度
                        }, 0) / cur.pointIds.length, //求交集占有率
                }),
                {} //最终得到{clusterId: 占有率}
            )
    );

/**
 * NOTE 方案2: 直接在brush的回调中，且search结束后，注册修改所有的TargetSelectionEntry
 */
/*
const tgtRegisterFunc = (selRetOrSingle: Record<string, boolean> | string) => {
    if (typeof selRetOrSingle === "object") {
        tgtEntryIds.forEach((id) => (db.nodesSelections[id] = selRetOrSingle));
    } else {
        //single
        tgtEntryIds.forEach((id) => {
            if (view.isShowAggregation) {
                const { pointIds } = view.clusters.find(
                    (d) => d.id === selRetOrSingle
                ) || { pointIds: [] };
                pointIds.forEach((d: string) => {
                    db.nodesSelections[id][d] = true;
                });
            } else {
                db.nodesSelections[id][selRetOrSingle] = true;
            }
        });
    }
};
*/
const curSrcEntryId = ref<Type_NodesSelectionEntryId>(srcEntryIds[0]);
const srcNodesDict = computed(() => db.nodesSelections[curSrcEntryId.value]);

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
    [srcNodesDict, tsneRet], //可能是重新计算的
    ([newDict, newTsneRet]) => {
        view.sourceCoords = newTsneRet.filter((d) => newDict[d.id]);
    },
    {
        deep: true,
        immediate: true,
        onTrigger(event) {
            console.warn(
                "in tsne,watch [srcNodesDict, tsneRet],update () => view.sourceCoords, triggered! ",
                event
            );
        },
    }
);

watch(
    () => view.sourceCoords,
    (newV) => {
        view.rescaledCoords = rescaleCoords(
            newV,
            [
                view.bodyMargins.left * view.bodyWidth,
                view.bodyWidth * (1 - view.bodyMargins.right),
            ],
            [
                view.bodyMargins.top * view.bodyHeight,
                view.bodyHeight * (1 - view.bodyMargins.bottom),
            ],
            (d: TsneCoord) => d.x,
            (d: TsneCoord) => d.y,
            db.name + "===" + props.viewName
        );
    },
    {
        immediate: true,
        deep: true,
    }
); //此watcher是为了： 在root和rerun的结果之间切换时，points坐标能根据tsneRet动态改变
////////////////// !SECTION points first calc
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION aggregated points
const arcFuncFunc = <
    T extends { id: string; x: number; y: number; mag: number },
    U extends d3.PieArcDatum<any>
>(
    d: T
) => d3.arc<U>().innerRadius(0).outerRadius(view.clusterRadiusScale(d.mag));
const outerArcs = computed(() =>
    view.aggregatedCoords.map((d) => {
        const arcPath = d3
            .arc<void>()
            .innerRadius(0)
            .outerRadius(view.clusterRadiusScale(d.mag) * 1.1)
            .startAngle(0)
            .endAngle(
                Number(renderAggregatedSelEntry.value[d.id] * 2 * Math.PI)
            )();

        return arcPath?.slice(0, arcPath.indexOf("L")) || "";
    })
);
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
//NOTE resize必须在brush和zoom之前，而brush和zoom谁前后无所谓
const { widthScaleRatio, heightScaleRatio } = useResize(
    // () => props.resizeEndSignal,
    () => view.resizeEndSignal,
    () => db.whenToRescaleMode === "onResizeEnd",
    // view.rescaledCoords,
    () => view.bodyWidth,
    () => view.bodyHeight,
    (): void => {
        view.rescaledCoords = rescaleCoords<TsneCoord, never>(
            view.sourceCoords as TsneCoord[],
            [
                view.bodyWidth * view.bodyMargins.left,
                view.bodyWidth * (1 - view.bodyMargins.right),
            ],
            [
                view.bodyHeight * view.bodyMargins.top,
                view.bodyHeight * (1 - view.bodyMargins.bottom),
            ],
            (d: TsneCoord) => d.x,
            (d: TsneCoord) => d.y,
            db.name + "===" + props.viewName
        );
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
                undefined, //extentFunction, using default
                thisDs.taskType === "node-classification"
                    ? newLabelType === "true"
                        ? trueLabels
                        : predLabels
                    : trueLabels,
                true,
                db.srcLinksArr
            );
        }
    },
    {
        immediate: true,
        deep: true,
    }
);
// NOTE // 在isShowAggregation是在Promise之后改变时：
// 分开两个写是可行的，因为两者不可能同时改变。
// 合并一起写需要用if判断isShowAggregation，这个值在switcher的Promise resolve之后才改变，这个时间点比coords改变要晚
//如果和上面的watch isShowAggregation合并写，会有循环修改依赖的问题
// NOTE 更正：现在isShowAggregation是直接改变了
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
                view.bodyWidth * view.bodyMargins.left,
                view.bodyHeight * view.bodyMargins.top,
            ],
            [
                view.bodyWidth * (1 - view.bodyMargins.right),
                view.bodyHeight * (1 - view.bodyMargins.bottom),
            ],
        ],
        // () => view.renderCoords,
        () =>
            view.isShowAggregation
                ? view.aggregatedCoords
                : view.rescaledCoords,
        () => {
            tgtEntryIds.forEach((entryId) => {
                db.nodesSelections[entryId] = {};
            });
            // localTargetEntry.value = {};
            // aggregatedLocalTargetEntry.value = {};
        },
        // tgtRegisterFunc({});
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
                            srcNodesDict.value[pointId];
                    });
                });
            } else {
                tgtEntryIds.forEach((entryId) => {
                    db.nodesSelections[entryId][id as Type_NodeId] =
                        srcNodesDict.value[id];
                });
            }
            // if (view.isShowAggregation)
            //     aggregatedLocalTargetEntry.value[id as Type_ClusterId] = true;
            // else localTargetEntry.value[id as Type_NodeId] = true;
        },
        () => true,
        () => db.clearSelMode === "manual",
        () => view.isBrushEnabled,
        (d) => d.x, //NOTE rescale之后的点是x，y获取
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

// console.log(" in tsne View, after useBrush and useZoom,  view is", view);

onMounted(() => {
    console.log("tsne renderer, mounted!");
    console.log(
        "tsne render, in mounted, width, height",
        view.bodyWidth,
        view.bodyHeight
    );
}); // onMounted END
onUnmounted(() => {
    console.log("tsne renderer, unmounted!");
});
</script>

<style scoped></style>
