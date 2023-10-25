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
                            db.highlightedNodeIds = srcGraphsDict[id]; //这里的每个id其实是graphId
                            db.highlightedWholeGraphIds[id] = true;
                        })
                    "
                    @mouseleave="
                        () => {
                            db.highlightedNodeIds = {};
                            db.highlightedWholeGraphIds = {};
                        }
                    "
                >
                    <g v-for="(arc, i) in d.arcs" :key="i">
                        <title>
                            {{
                                (db.labelType === "true" ? "true" : "pred") +
                                `label: ${arc.data[0]}\ncount: ${arc.data[1]}`
                            }}
                        </title>
                        <path
                            :fill="colorScale(arc.data[0])"
                            :d="arcFuncFunc(d)(arc) || ''"
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

            <template v-else>
                <g class="points" stroke-width="1.5">
                    <template v-for="d in view.rescaledCoords" :key="d.id">
                        <g :transform="`translate(${d.x} ${d.y})`">
                            <path
                                class="outer"
                                :d="circlePath(view.nodeRadius)"
                                :fill="
                                    isDbWiseComparativeDb(db)
                                        ? 'white'
                                        : colorScale(curLabels[+d.id] + '')
                                "
                                :stroke="
                                    renderSingleSelEntry[d.id]
                                        ? 'black'
                                        : colorScale(curLabels[+d.id] + '')
                                "
                                :stroke-width="1"
                                :opacity="dynamicOpacity(d.id)"
                                @mouseenter="
                                    () => {
                                        db.highlightedWholeGraphIds[
                                            d.id
                                        ] = true;
                                        db.highlightedNodeIds =
                                            srcGraphsDict[d.id];
                                    }
                                "
                                @mouseleave="
                                    () => {
                                        db.highlightedNodeIds = {};
                                        db.highlightedWholeGraphIds = {};
                                    }
                                "
                            >
                                <title>
                                    {{
                                        !isDbWiseComparativeDb(db)
                                            ? `gid: ${d.id}\n` +
                                              (db.labelType === "true"
                                                  ? "true"
                                                  : "pred") +
                                              `Label: ${curLabels[+d.id]}`
                                            : `gid: ${d.id}\n` +
                                              (db.labelType === "true"
                                                  ? "true"
                                                  : "pred") +
                                              `Label: ${
                                                  curLabels[+d.id]
                                              }\nnode count: ${(
                                                  graphMapParentDb0Percent(
                                                      d.id
                                                  ) * 100
                                              ).toFixed(1)}% from db0 VS. ${(
                                                  graphMapParentDb1Percent(
                                                      d.id
                                                  ) * 100
                                              ).toFixed(1)}% from db1`
                                    }}
                                </title>
                            </path>

                            <path
                                v-if="isDbWiseComparativeDb(db)"
                                class="L"
                                :d="
                                    leftPercentCirclePath(
                                        view.nodeRadius,
                                        graphMapParentDb0Percent(d.id) / 2
                                    )
                                "
                                stroke-linejoin="bevel"
                                :fill="colorScale(curLabels[+d.id] + '')"
                                :stroke="
                                    renderSingleSelEntry[d.id]
                                        ? 'black'
                                        : 'none'
                                "
                                :stroke-width="1"
                                :opacity="dynamicOpacity(d.id)"
                                @mouseenter="
                                    () => {
                                        db.highlightedWholeGraphIds[
                                            d.id
                                        ] = true;
                                        db.highlightedNodeIds =
                                            srcGraphsDict[d.id];
                                    }
                                "
                                @mouseleave="
                                    () => {
                                        db.highlightedNodeIds = {};
                                        db.highlightedWholeGraphIds = {};
                                    }
                                "
                            >
                                <title>
                                    {{
                                        `gid: ${d.id}\n` +
                                        (db.labelType === "pred"
                                            ? "pred"
                                            : "true") +
                                        `Label: ${
                                            curLabels[+d.id]
                                        }\nnode count: ${(
                                            graphMapParentDb0Percent(d.id) * 100
                                        ).toFixed(1)}% from db0 VS. ${(
                                            graphMapParentDb1Percent(d.id) * 100
                                        ).toFixed(1)}% from db1`
                                    }}
                                </title>
                            </path>
                            <path
                                v-if="isDbWiseComparativeDb(db)"
                                class="R"
                                :d="
                                    rightPercentCirclePath(
                                        view.nodeRadius,
                                        graphMapParentDb1Percent(d.id) / 2
                                    )
                                "
                                stroke-linejoin="bevel"
                                :fill="colorScale(curLabels[+d.id] + '')"
                                :stroke="
                                    renderSingleSelEntry[d.id]
                                        ? 'black'
                                        : 'none'
                                "
                                :stroke-width="1"
                                :opacity="dynamicOpacity(d.id)"
                                @mouseenter="
                                    () => {
                                        db.highlightedWholeGraphIds[
                                            d.id
                                        ] = true;
                                        db.highlightedNodeIds =
                                            srcGraphsDict[d.id];
                                    }
                                "
                                @mouseleave="
                                    () => {
                                        db.highlightedNodeIds = {};
                                        db.highlightedWholeGraphIds = {};
                                    }
                                "
                            >
                                <title>
                                    {{
                                        `gid: ${d.id}\n` +
                                        (db.labelType === "pred"
                                            ? "pred"
                                            : "true") +
                                        `Label: ${
                                            curLabels[+d.id]
                                        }\nnode count: ${(
                                            graphMapParentDb0Percent(d.id) * 100
                                        ).toFixed(1)}% from db0 VS. ${(
                                            graphMapParentDb1Percent(d.id) * 100
                                        ).toFixed(1)}% from db1`
                                    }}
                                </title>
                            </path>
                        </g>
                    </template>
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
import {
    isEmptyDict,
    nodeMapGraph2GraphMapNodes,
    rescaleCoords,
} from "../../utils/graphUtils";
import type {
    AggregatedView,
    GraphTsneCoord,
    LinkableView,
    Type_ClusterId,
    Type_GraphId,
    Type_NodeId,
} from "../../types/types";
import { useMyStore } from "../../stores/store";
import * as d3 from "d3";
import type {
    CompDashboard,
    SingleDashboard,
    Type_NodesSelectionEntryId,
} from "@/types/types";
import { useD3Brush } from "../plugin/useD3Brush";
import { useD3Zoom } from "../plugin/useD3Zoom";
import { useResize } from "../plugin/useResize";
import { isDbWiseComparativeDb } from "@/types/typeFuncs";
import {
    circlePath,
    rightPercentCirclePath,
    leftPercentCirclePath,
} from "@/utils/otherUtils";

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
        ? undefined
        : props.which == 1
        ? myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[1])
        : myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[1]);
const view: AggregatedView & LinkableView = myStore.getViewByName(
    db,
    props.viewName
) as AggregatedView & LinkableView;
console.log("in db", db, " in graph tsne View, view is", view);

const predLabels = thisDs.graphPredLabels || [];
const trueLabels = thisDs.graphTrueLabels || []; //we assume both datasets should have graphTrueLabels
const curLabels = computed(() =>
    db.labelType === "true" ? trueLabels : predLabels
);

const colorScale = computed(() => thisDs.graphColorScale || (() => "#888"));

const highlightedGraphIdsByNodes = computed(() => {
    return nodeMapGraph2GraphMapNodes(
        db.highlightedNodeIds as (typeof db.nodesSelections)[string]
    );
});
const dynamicOpacity = computed(
    () => (i: Type_GraphId) =>
        !db.isHighlightCorrespondingNode || isEmptyDict(db.highlightedNodeIds)
            ? 1
            : highlightedGraphIdsByNodes.value[i] ||
              db.highlightedWholeGraphIds[i]
            ? 1
            : 0.1
);
const graphTsneRet = computed(() =>
    props.which == 1
        ? (db as CompDashboard).graphTsneRet1
        : props.which == 2
        ? (db as CompDashboard).graphTsneRet2
        : (db as SingleDashboard).graphTsneRet || []
);
////////////////// !SECTION data prepare
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION nodeSelectionEntry related
const srcEntryIds = myStore.getViewSourceNodesSelectionEntry(props.viewName);
const curSrcEntryId = ref<Type_NodesSelectionEntryId>(srcEntryIds[0]);
const srcNodesDict = computed(() => db.nodesSelections[curSrcEntryId.value]);
const srcGraphsDict = computed(() =>
    nodeMapGraph2GraphMapNodes(db.nodesSelections[curSrcEntryId.value])
);
const graphMapParentDb1Percent = (gid: Type_GraphId) => {
    const nodeInfos = Object.values(srcGraphsDict.value[gid]);
    const fromBoth = nodeInfos.filter((d) => d.parentDbIndex === 2);
    const from1 = nodeInfos.filter((d) => d.parentDbIndex === 1);
    // if (fromBoth.length === nodeInfos.length) {
    //     return 1;
    // }
    return (from1.length + fromBoth.length) / nodeInfos.length;
    // NOTE a/(a+b) === (a+c)/(a+b+c+c)?
};
const graphMapParentDb0Percent = (gid: Type_GraphId) => {
    const nodeInfos = Object.values(srcGraphsDict.value[gid]);
    const fromBoth = nodeInfos.filter((d) => d.parentDbIndex === 2);
    const from0 = nodeInfos.filter((d) => d.parentDbIndex === 0);
    return (from0.length + fromBoth.length) / nodeInfos.length;
    // NOTE a/(a+b) === (a+c)/(a+b+c+c)?
};

const tgtEntryIds = myStore.getViewTargetNodesSelectionEntry(props.viewName);
const selectedGraphIdsByNodes = computed(() => {
    return nodeMapGraph2GraphMapNodes(
        db.nodesSelections[tgtEntryIds[0]] as Record<
            string,
            {
                gid: string;
                parentDbIndex: number;
                hop: number;
            }
        >
    ); //REVIEW why 0?
});
const renderSingleSelEntry = computed(() => selectedGraphIdsByNodes.value);
const renderAggregatedSelEntry: ComputedRef<Record<Type_NodeId, number>> =
    computed(() =>
        view.clusters.reduce(
            (acc, cur) => ({
                ...acc,
                [cur.id]:
                    cur.pointIds.reduce((inAcc, inCur) => {
                        if (selectedGraphIdsByNodes.value[inCur]) {
                            inAcc += Object.values(
                                selectedGraphIdsByNodes.value[inCur]
                            ).length;
                        }
                        return inAcc; //求交集长度
                    }, 0) /
                    cur.pointIds.reduce((inAcc, inCur) => {
                        inAcc += Object.keys(srcGraphsDict.value[inCur]).length;
                        return inAcc;
                    }, 0), //求交集占有率
            }),
            {} //最终得到{clusterId: 占有率}
        )
    );

////////////////// !SECTION nodeSelectionEntry related
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION points first calc
watch(
    [srcGraphsDict, graphTsneRet], //可能是重新计算的
    ([newDict, newTsneRet]) => {
        view.sourceCoords = newTsneRet.filter((d) => newDict[d.id]);
    },
    {
        deep: true,
        immediate: true,
        onTrigger(event) {
            console.warn(
                "in graph tsne, watch [srcNodesDict, graphTsneRet],update () => view.sourceCoords, triggered! ",
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
            (d: GraphTsneCoord) => d.x,
            (d: GraphTsneCoord) => d.y,
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
const outerArcFunc = <
    T extends {
        id: Type_GraphId;
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
        view.rescaledCoords = rescaleCoords<GraphTsneCoord, never>(
            view.sourceCoords as GraphTsneCoord[],
            [
                view.bodyWidth * view.bodyMargins.left,
                view.bodyWidth * (1 - view.bodyMargins.right),
            ],
            [
                view.bodyHeight * view.bodyMargins.top,
                view.bodyHeight * (1 - view.bodyMargins.bottom),
            ],
            (d: GraphTsneCoord) => d.x,
            (d: GraphTsneCoord) => d.y,
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
                newLabelType === "true" ? trueLabels : predLabels,
                false,
                []
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
        },
        (id: Type_GraphId | Type_ClusterId) => {
            db.fromViewName = props.viewName;
            if (view.isShowAggregation) {
                const { pointIds: graphIds } = view.clusters.find(
                    (d) => d.id === id
                ) || {
                    pointIds: [],
                };
                graphIds.forEach((graphId) => {
                    tgtEntryIds.forEach((entryId) => {
                        db.nodesSelections[entryId] = {
                            ...db.nodesSelections[entryId],
                            ...srcGraphsDict.value[graphId],
                        };
                    });
                });
            } else {
                tgtEntryIds.forEach((entryId) => {
                    db.nodesSelections[entryId] = {
                        ...db.nodesSelections[entryId],
                        ...srcGraphsDict.value[id],
                    };
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

// console.log(" in graph tsne View, after useBrush and useZoom,  view is", view);

onMounted(() => {
    console.log("graph tsne renderer, mounted!");
    console.log(
        "graph tsne render, in mounted, width, height",
        view.bodyWidth,
        view.bodyHeight
    );
}); // onMounted END
onUnmounted(() => {
    console.log("graph tsne renderer, unmounted!");
});
</script>

<style scoped></style>
