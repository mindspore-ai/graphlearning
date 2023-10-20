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
            id="margin"
            :transform="`translate(${mL} ${mT})`"
            :height="view.bodyHeight - mT - mB"
            :width="view.bodyWidth - mL - mR"
        >
            <g
                id="title"
                :height="titleSize"
                :width="matrixSize"
                :transform="`translate(${axisSize} 0)`"
            >
                <text
                    fill="black"
                    text-anchor="middle"
                    :x="matrixSize / 2"
                    :font-size="titleSize * 0.8"
                    :dy="titleSize * 0.4"
                >
                    Confusion Matrix
                </text>
            </g>

            <g
                id="yAxisName"
                :transform="`translate(${view.bodyWidth * 0.01} ${
                    view.bodyHeight * 0.07
                })`"
            >
                <text>true labels</text>
            </g>
            <g
                id="yAxis"
                :height="matrixSize"
                :width="axisSize"
                :transform="`translate(${axisSize} ${titleSize})`"
            >
                <g id="yAxisTicks" font-size="1em" text-anchor="end">
                    <g
                        v-for="(d, i) in labelData"
                        :key="i"
                        :transform="`translate(0 ${
                            axisScale(d)! + axisScale.bandwidth() / 2 })`"
                    >
                        <line stroke="currentColor" :x2="-6"></line>
                        <text
                            :fill="labelColorScale(i + '')"
                            :x="-7"
                            dy="0.3em"
                        >
                            {{ d }}
                        </text>
                    </g>
                </g>
            </g>

            <g
                id="xAxisName"
                :transform="`translate(${view.bodyWidth * 0.7} ${
                    view.bodyHeight * 0.9
                })`"
            >
                <text>pred labels</text>
            </g>
            <g
                id="xAxis"
                :height="axisSize"
                :width="matrixSize"
                :transform="`translate(${axisSize} ${titleSize + matrixSize})`"
            >
                <g id="xAxisTicks" font-size="1em" text-anchor="start">
                    <g
                        v-for="(d, i) in labelData"
                        :key="i"
                        :transform="`translate(${
                            axisScale(d)! + axisScale.bandwidth() / 2
                        },0)`"
                    >
                        <line stroke="currentColor" :y2="6"></line>
                        <text
                            :fill="labelColorScale(i + '')"
                            :y="15"
                            dy="0.3em"
                            transform="rotate(45)"
                        >
                            {{ d }}
                        </text>
                    </g>
                </g>
            </g>

            <g
                id="legend"
                :height="matrixSize"
                :width="legendSize * 0.9"
                :transform="`translate(${
                    axisSize + matrixSize + legendSize * 0.1
                } ${titleSize})`"
            >
                <svg :height="matrixSize" :width="legendSize * 0.9"></svg>
                <linearGradient
                    :id="`gradient-${db.id}-${which}`"
                    x1="100%"
                    y1="0%"
                    x2="100%"
                    y2="100%"
                >
                    <stop
                        v-for="i in 9"
                        :key="i"
                        :offset="`${((i - 1) / 8) * 100}%`"
                        :stop-color="d3.interpolateBlues(1 - (i - 1) / 8)"
                    ></stop>
                </linearGradient>
                <rect
                    :height="matrixSize"
                    :width="legendSize * 0.9"
                    stroke-width="1"
                    stroke="black"
                    :fill="`url(#gradient-${db.id}-${which})`"
                ></rect>
            </g>

            <g
                id="legendAxis"
                :height="matrixSize"
                :width="axisSize"
                :transform="`translate(${
                    axisSize + matrixSize + legendSize
                } ${titleSize})`"
            >
                <g id="legendAxisTicks" font-size="1em" text-anchor="start">
                    <g
                        v-for="(d, i) in legendTicks"
                        :key="i"
                        :transform="`translate(0, ${legendAxisScale(d)})`"
                    >
                        <line :x2="6" fill="currentColor"></line>
                        <text :dx="8" dy="0.3em">{{ d }}</text>
                    </g>
                </g>
            </g>

            <g
                id="matrix"
                :transform="`translate(${axisSize} ${titleSize})`"
                :height="matrixSize"
                :width="matrixSize"
            >
                <rect
                    fill="none"
                    stroke="#000"
                    stroke-width="1"
                    x="0"
                    y="0"
                    :height="matrixSize"
                    :width="matrixSize"
                ></rect>
                <!--vue中的template 若v-for一个纯数字，i从1开始-->
                <g
                    class="row"
                    v-for="i in numClasses"
                    :key="i"
                    :transform="`translate(0,${yScale(i - 1)})`"
                >
                    <g
                        class="cell"
                        v-for="j in numClasses"
                        :key="j"
                        :transform="`translate(${xScale(j - 1)}, 0)`"
                        :height="yScale.bandwidth()"
                        :width="xScale.bandwidth()"
                        @mouseenter="
                            () => {
                                db.highlightedNodeIds =
                                    confusionMatrix[i - 1][j - 1].nodesDict;

                                // if (taskType === 'graph-classification')//不加亦可。
                                db.highlightedWholeGraphIds =
                                    confusionMatrix[i - 1][j - 1].graphsDict;
                            }
                        "
                        @mouseleave="
                            () => {
                                db.highlightedNodeIds = {};
                                // if (taskType === 'graph-classification')//不加亦可
                                db.highlightedWholeGraphIds = {};
                            }
                        "
                    >
                        <g v-if="!isDbWiseComparativeDb(db)">
                            <rect
                                :x="0"
                                :y="0"
                                :height="yScale.bandwidth()"
                                :width="xScale.bandwidth()"
                                stroke-width="0.5"
                                stroke="rgb(198, 198, 198)"
                                :fill="
                                    matrixColorScale(
                                        confusionMatrix[i - 1][j - 1].count as number
                                    )
                                "
                            ></rect>
                            <text
                                text-anchor="middle"
                                font-size="12"
                                :x="xScale.bandwidth() / 2"
                                :y="2 + yScale.bandwidth() / 2"
                                :fill="
                                    (confusionMatrix[i - 1][j - 1].count as number) >
                                    (d3.mean(
                                        matrixColorScale.domain(),
                                        (d) => d
                                    ) || Infinity)
                                        ? 'white'
                                        : 'black'
                                "
                            >
                                {{ confusionMatrix[i - 1][j - 1].count }}
                            </text>
                        </g>
                        <g v-else>
                            <rect
                                :x="0"
                                :y="0"
                                :height="yScale.bandwidth()"
                                :width="xScale.bandwidth()"
                                stroke-width="0.5"
                                stroke="rgb(198, 198, 198)"
                                :fill-opacity="0"
                            ></rect>
                            <g
                                :transform="`translate(${
                                    xScale.bandwidth() / 2
                                } ${yScale.bandwidth() / 2})`"
                            >
                                <path
                                    :d="
                                        leftPercentCirclePath(
                                            Math.min(
                                                yScale.bandwidth(),
                                                xScale.bandwidth()
                                            ) / 2,
                                            0.5
                                        )
                                    "
                                    stroke="rgb(198,198,198)"
                                    stroke-width="0.5"
                                    :fill=" matrixColorScale( (confusionMatrix[i - 1][j - 1].count as [number,number])[0])
                                    "
                                ></path>

                                <path
                                    :d="
                                        rightPercentCirclePath(
                                            Math.min(
                                                yScale.bandwidth(),
                                                xScale.bandwidth()
                                            ) / 2,
                                            0.5
                                        )
                                    "
                                    stroke="rgb(198,198,198)"
                                    stroke-width="0.5"
                                    :fill="
                                        matrixColorScale(
                                            (confusionMatrix[i - 1][j - 1]
                                                .count as [number,number])[1]
                                                 
                                        )
                                    "
                                ></path>
                            </g>
                            <text
                                text-anchor="middle"
                                font-size="12"
                                :x="xScale.bandwidth() / 4"
                                :y="2 + yScale.bandwidth() / 2"
                                :fill="
                                    (confusionMatrix[i - 1][j - 1].count as [number,number])[0] >
                                    (d3.mean(
                                        matrixColorScale.domain(),
                                        (d) => d
                                    ) || Infinity)
                                        ? 'white'
                                        : 'black'
                                "
                            >
                                {{
                                    (
                                        confusionMatrix[i - 1][j - 1].count as [
                                            number,
                                            number
                                        ]
                                    )[0]
                                }}
                            </text>
                            <text
                                text-anchor="middle"
                                font-size="12"
                                :x="(xScale.bandwidth() * 3) / 4"
                                :y="2 + yScale.bandwidth() / 2"
                                :fill="
                                    (confusionMatrix[i - 1][j - 1].count as [number,number])[1] >
                                    (d3.mean(
                                        matrixColorScale.domain(),
                                        (d) => d
                                    ) || Infinity)
                                        ? 'white'
                                        : 'black'
                                "
                            >
                                {{
                                    (
                                        confusionMatrix[i - 1][j - 1].count as [
                                            number,
                                            number
                                        ]
                                    )[1]
                                }}
                            </text>
                        </g>
                    </g>
                </g>
                <g class="gBrush" ref="brushRef"></g>
            </g>
        </g>
    </svg>
</template>
<script setup lang="ts">
/** @description
 * This view is for node-classification and graph-classification
 * For comparative dashboard, there will be 2 views.
 */
import { useMyStore } from "@/stores/store";
import * as d3 from "d3";
import type {
    CompDashboard,
    Dataset,
    NodeCoord,
    SingleDashboard,
    TsneCoord,
    Type_GraphId,
    Type_NodeId,
    Type_NodesSelectionEntryId,
} from "@/types/types";
import {
    computed,
    watch,
    ref,
    onMounted,
    onUnmounted,
    onBeforeUnmount,
} from "vue";
import { useResize } from "../plugin/useResize";
import {
    nodeMapGraph2GraphMapNodes,
    graphMapNodes2nodeMapGraph,
} from "@/utils/graphUtils";
import { isDbWiseComparativeDb } from "@/types/typeFuncs";
import {
    leftPercentCirclePath,
    rightPercentCirclePath,
} from "@/utils/otherUtils";
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
    props.which === 0
        ? undefined
        : props.which === 1
        ? myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[1])
        : myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[0]);
const view = myStore.getViewByName(db, props.viewName)!;

const { taskType } = thisDs;
const predLabels =
    taskType === "node-classification"
        ? thisDs.predLabels || []
        : thisDs.graphPredLabels || [];
const trueLabels =
    taskType === "node-classification"
        ? thisDs.trueLabels || theOtherDs?.trueLabels || []
        : thisDs.graphTrueLabels || []; //we assume both datasets should have graphTrueLabels
const numClasses =
    taskType === "node-classification"
        ? thisDs.numNodeClasses || theOtherDs?.numNodeClasses || 0
        : thisDs.numGraphClasses || theOtherDs?.numGraphClasses || 0;
//NOTE 如果原始文件没有，通过set计算的过程在fetchOriginDataset里

const labelData = Array.from({ length: numClasses }, (v, i) => `class ${i}`); //REVIEW

const srcNodeEntry = myStore.getViewSourceNodesSelectionEntry(props.viewName);
const tgtNodeEntry = myStore.getViewTargetNodesSelectionEntry(props.viewName);

const curSrcNodeEntry = ref<Type_NodesSelectionEntryId>(srcNodeEntry[0]);
const srcNodesDict = computed(() => db.nodesSelections[curSrcNodeEntry.value]);

const getIdByLabelIndex =
    taskType === "node-classification"
        ? (index: number) => index + ""
        : (index: number) => thisDs.graphIndex[index];
const globalTrueLabelsWithIds = trueLabels.map((d, i) => ({
    id: getIdByLabelIndex(i),
    label: d,
}));
const globalPredLabelsWithIds = predLabels.map((d, i) => ({
    id: getIdByLabelIndex(i),
    label: d,
}));

const isGraphContainsNode = (graphId: Type_GraphId, nodeId: Type_NodeId) =>
    thisDs.graphRecords[graphId].nodesRecord[nodeId];

const graphMapNodes = computed(() =>
    nodeMapGraph2GraphMapNodes(srcNodesDict.value)
);

const graphMapParentDbIndex = (gid: Type_GraphId) => {
    let has0 = false;
    let has1 = false;
    for (const nodeInfo of Object.values(graphMapNodes.value[gid])) {
        if (nodeInfo.parentDbIndex === 2) return 2;

        if (nodeInfo.parentDbIndex === 0) has0 = true;
        if (nodeInfo.parentDbIndex === 1) has1 = true;
        if (has0 && has1) return 2;
    }
    if (has0) return 0;
    if (has1) return 1;
};

const currentTrueLabels = globalTrueLabelsWithIds.filter(
    taskType === "node-classification"
        ? ({ id: nodeId, label: nodeLabel }) => srcNodesDict.value[nodeId]
        : ({ id: graphId, label: graphLabel }) => graphMapNodes.value[graphId]
);
const currentPredLabels = globalPredLabelsWithIds.filter(
    taskType === "node-classification"
        ? ({ id: nodeId, label: nodeLabel }) => srcNodesDict.value[nodeId]
        : ({ id: graphId, label: graphLabel }) => graphMapNodes.value[graphId]
);
const confusionMatrix = computed(() => {
    const confusionMatrix = Array.from({ length: numClasses }, () =>
        Array.from({ length: numClasses }, () => ({
            count: isDbWiseComparativeDb(db) ? ([0, 0] as [number, number]) : 0,
            nodesDict: {} as Record<
                Type_NodeId,
                {
                    gid: Type_GraphId; //the graph that the node is affiliated to
                    parentDbIndex: number; //usually 0 or 1;
                    //   hop: number; //0: step1Neighbor, 1: step2Neighbor, -1:originSelf
                }
            >,
            graphsDict: {} as Record<
                Type_GraphId,
                Record<
                    Type_NodeId,
                    {
                        gid: Type_GraphId; //the graph that the node is affiliated to
                        parentDbIndex: number; //usually 0 or 1;
                        //   hop: number; //0: step1Neighbor, 1: step2Neighbor, -1:originSelf
                    }
                >
            >,
        }))
    );

    // 填充混淆矩阵
    for (let i = 0; i < currentPredLabels.length; i++) {
        const trueLabel = currentTrueLabels[i].label;
        const predLabel = currentPredLabels[i].label;
        if (!isDbWiseComparativeDb(db)) {
            (confusionMatrix[trueLabel][predLabel].count as number)++; //NOTE 从0开始
            if (taskType === "node-classification") {
                const curNodeId = currentTrueLabels[i].id;
                const curGraphId = srcNodesDict.value[curNodeId];
                confusionMatrix[trueLabel][predLabel].nodesDict[curNodeId] =
                    curGraphId;
                // NOTE omit graphsDict
            } else {
                const curGraphId = currentTrueLabels[i].id;
                confusionMatrix[trueLabel][predLabel].graphsDict[curGraphId] =
                    graphMapNodes.value[curGraphId];
                confusionMatrix[trueLabel][predLabel].nodesDict =
                    graphMapNodes2nodeMapGraph(
                        confusionMatrix[trueLabel][predLabel].graphsDict
                    );
            }
        } else {
            if (taskType === "node-classification") {
                const count = confusionMatrix[trueLabel][predLabel].count as [
                    number,
                    number
                ];
                const curNodeId = currentTrueLabels[i].id;
                const graphInfo = srcNodesDict.value[curNodeId];
                const { parentDbIndex } = graphInfo as {
                    gid: string;
                    parentDbIndex: number;
                    hop: number;
                };
                if (parentDbIndex !== 2) {
                    count[parentDbIndex]++;
                } else {
                    count[0]++;
                    count[1]++;
                }

                confusionMatrix[trueLabel][predLabel].nodesDict[curNodeId] =
                    graphInfo;
                // NOTE omit graphsDict
            } else {
                const count = confusionMatrix[trueLabel][predLabel].count as [
                    number,
                    number
                ];
                const curGraphId = currentTrueLabels[i].id;
                confusionMatrix[trueLabel][predLabel].graphsDict[curGraphId] =
                    graphMapNodes.value[curGraphId];
                confusionMatrix[trueLabel][predLabel].nodesDict =
                    graphMapNodes2nodeMapGraph(
                        confusionMatrix[trueLabel][predLabel].graphsDict
                    );
                const parentDbIndex = graphMapParentDbIndex(curGraphId);
                // console.log(
                //     "in calc confusion matrix of dbWiseComp of graph-classification, got a parentDbIndex of gid ",
                //     curGraphId,
                //     ":",
                //     parentDbIndex
                // );
                switch (parentDbIndex) {
                    case 0:
                        count[0]++;
                        break;
                    case 1:
                        count[1]++;
                        break;
                    case 2:
                        count[0]++;
                        count[1]++;
                        break;
                    default:
                        break;
                }
            }
        }
    }

    return confusionMatrix;
});
////////////////// !SECTION data prepare
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION sizes & scales
const axisSize = computed(
    () => 0.06 * Math.min(view.bodyHeight, view.bodyWidth)
);
const titleSize = computed(() => 0.08 * view.bodyHeight);
const legendSize = computed(() => 0.1 * view.bodyWidth); //include the margin between right border of matrix and legend
const mL = computed(() => view.bodyWidth * view.bodyMargins.left);
const mR = computed(() => view.bodyWidth * view.bodyMargins.right);
const mT = computed(() => view.bodyHeight * view.bodyMargins.top);
const mB = computed(() => view.bodyHeight * view.bodyMargins.bottom);
const matrixSize = computed(() =>
    Math.min(
        view.bodyHeight -
            mT.value -
            titleSize.value -
            axisSize.value -
            mB.value,
        view.bodyWidth -
            mL.value -
            2 * axisSize.value -
            legendSize.value -
            mR.value
    )
);
const matrixColorScale = d3
    .scaleSequential(d3.interpolateBlues)
    .domain([
        0,
        d3.max<number>(
            confusionMatrix.value.flatMap((d) => d.flatMap((dd) => dd.count))
        ) as number,
    ]); //NOTE flatMap对于count是数组和数字都适用

const legendAxisScale = computed(() =>
    d3
        .scaleLinear()
        .domain([
            0,
            d3.max<number>(
                confusionMatrix.value.flatMap((d) =>
                    d.flatMap((dd) => dd.count)
                )
            ) as number,
        ]) //NOTE flatMap对于count是数组和数字都适用
        .range([matrixSize.value, 0])
);
const numLegendTicks = 11;
const legendTicks = computed(() => legendAxisScale.value.ticks(numLegendTicks));
const xScale = computed(() =>
    d3
        .scaleBand<number>()
        .domain(d3.range(numClasses))
        .range([0, matrixSize.value])
);
const yScale = computed(() =>
    d3
        .scaleBand<number>()
        .domain(d3.range(numClasses))
        .range([0, matrixSize.value])
);
const axisScale = computed(() =>
    d3
        .scaleBand<string>()
        .domain(labelData)
        .range([0, matrixSize.value])
        .align(0.5)
);

const labelColorScale =
    taskType === "node-classification"
        ? thisDs.colorScale || theOtherDs?.colorScale || (() => "black")
        : thisDs.graphColorScale ||
          theOtherDs?.graphColorScale ||
          (() => "black");
///////////////// !SECTION sizes & scales
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
///////////////// SECTION brush
const brushRef = ref<SVGGElement | null>(null);
const brush = d3.brush().extent([
    [0, 0],
    [matrixSize.value, matrixSize.value],
]);

const brushStart = (e: d3.D3BrushEvent<void>) => {
    if (!e.sourceEvent) return;
    console.log("in ", props.viewName, "brushStart");
    if (db.clearSelMode === "auto") {
        tgtNodeEntry.forEach((entryId) => {
            db.nodesSelections[entryId] = {};
        });
    }
};

const localSearchFlag = ref(true);
const brushEnd = (e: d3.D3BrushEvent<void>) => {
    if (!e.sourceEvent || !localSearchFlag.value) return;
    const ext = e.selection;
    if (ext) {
        const [[x0, y0], [x1, y1]] = ext as [
            [number, number],
            [number, number]
        ];
        const startRowIndex = Math.floor(y0 / yScale.value.bandwidth());
        const endRowIndex = Math.ceil(y1 / yScale.value.bandwidth()) - 1;
        const startColIndex = Math.floor(x0 / xScale.value.bandwidth());
        const endColIndex = Math.ceil(x1 / xScale.value.bandwidth()) - 1;

        const expandedX0 = startColIndex * xScale.value.bandwidth();
        const expandedX1 = (endColIndex + 1) * xScale.value.bandwidth();
        const expandedY0 = startRowIndex * yScale.value.bandwidth();
        const expandedY1 = (endRowIndex + 1) * yScale.value.bandwidth();

        // brush.on("start", null).on("end", null);//NOTE 有了事件非空判断，不用取消绑定也行了
        const gBrush = d3.select(brushRef.value);
        if (gBrush.node()) {
            (gBrush as d3.Selection<SVGGElement, unknown, null, undefined>)
                .transition()
                .delay(100)
                .duration(ext ? 750 : 0)
                .call(brush.move, [
                    [expandedX0, expandedY0],
                    [expandedX1, expandedY1],
                ]);
        }
        // brush.on("start", brushStart).on("end", brushEnd); //NOTE 有了事件非空判断，不用取消绑定也行了

        let localSumDict = {};
        for (let i = startRowIndex; i <= endRowIndex; i++) {
            for (let j = startColIndex; j <= endColIndex; j++) {
                localSumDict = {
                    ...confusionMatrix.value[i][j].nodesDict,
                    ...localSumDict,
                };
            }
        }
        db.fromViewName = props.viewName;
        tgtNodeEntry.forEach((entryId) => {
            db.nodesSelections[entryId] = {
                ...localSumDict,
                ...db.nodesSelections[entryId],
            };
        });

        // localSearchFlag.value = true;
    }
};
brush.on("start", brushStart).on("end", brushEnd);

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION resize
//NOTE resize必须在brush和zoom之前，而brush和zoom谁前后无所谓
const { widthScaleRatio, heightScaleRatio } = useResize(
    // () => props.resizeEndSignal,
    () => view.resizeEndSignal,
    () => db.whenToRescaleMode === "onResizeEnd",
    // view.rescaledCoords,
    matrixSize,
    matrixSize,
    (): void => {},
    db.name + "===" + props.viewName
);
////////////////// !SECTION resize
////////////////////////////////////////////////////////////////////////////////
watch([widthScaleRatio, heightScaleRatio], ([newWR, newHR]) => {
    const gBrush = d3.select(brushRef.value);

    if (gBrush.node()) {
        const rect = d3.brushSelection(gBrush.node() as SVGGElement);

        if (rect) {
            const [[x0, y0], [x1, y1]] = rect as [
                [number, number],
                [number, number]
            ];

            (
                gBrush as d3.Selection<SVGGElement, unknown, null, undefined>
            ).call(brush.move, () => {
                if (newWR != Infinity && newHR != Infinity) {
                    return [
                        [x0 * newWR, y0 * newHR],
                        [x1 * newWR, y1 * newHR],
                    ];
                } else {
                    return [
                        //if Infinity, then 0 * Inf = NaN, then there'll be bugs, esp. when switch dashboards
                        [0, 0],
                        [0, 0],
                    ];
                }
            });
        }
    }
});

onMounted(() => {
    view.brushEnableFunc = ref(() => {
        const gBrush = d3.select(brushRef.value);
        if (gBrush.node()) {
            gBrush.style("display", "inherit");
            gBrush.attr("pointer-events", "all");

            (
                gBrush as d3.Selection<SVGGElement, unknown, null, undefined>
            ).call(brush); // rebind
        }
    });
    view.brushDisableFunc = ref(() => {
        const gBrush = d3.select(brushRef.value);
        if (gBrush.node()) {
            gBrush.attr("pointer-events", "none");
            gBrush.style("display", "none");
        }
    });
    view.hideRectWhenClearSelFunc = ref(() => {
        const gBrush = d3.select(brushRef.value);
        if (gBrush.node()) {
            if (d3.brushSelection(gBrush.node() as SVGGElement)) {
                // NOTE it's not enough to only check whether it's in brush mode
                //we should also check whether the rect exists
                (
                    gBrush as d3.Selection<
                        SVGGElement,
                        unknown,
                        null,
                        undefined
                    >
                ).call(brush.clear);
            }
        }
    });

    if (view.isBrushEnabled) view.brushEnableFunc();
});
onBeforeUnmount(() => {
    view.hideRectWhenClearSelFunc();
});
onUnmounted(() => {
    const gBrush = d3.select(brushRef.value);
    if (gBrush && gBrush.node()) {
        gBrush.remove();
    }
    view.brushEnableFunc = () => {};
    view.brushDisableFunc = () => {};
    view.hideRectWhenClearSelFunc = () => {};
});
///////////////// !SECTION brush
////////////////////////////////////////////////////////////////////////////////
</script>

<style scoped></style>
