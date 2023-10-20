<template>
    <el-scrollbar
        :max-height="view.bodyHeight"
        :view-style="{
            display: 'flex',
            flexDirection: 'column',
            height: view.bodyHeight + 'px',
        }"
    >
        <div :style="{ textAlign: 'center' }">all nodes features</div>
        <svg :width="sumChartWidth" :height="sumChartHeight" id="sumSvg">
            <g
                id="sumYAxis"
                :transform="`translate(${
                    sumChartYAxisWidth + sumChartMarginLeft
                } ${0})`"
            >
                <g
                    id="sumYAxisTicks"
                    :font-size="
                        ((sumChartHeight -
                            sumChartMarginTop -
                            sumChartMarginBottom -
                            xAxisHeight) *
                            0.8) /
                        sumChartYTickNum
                    "
                    text-anchor="end"
                >
                    <g
                        v-for="d in sumChartYTicks"
                        :key="d"
                        :transform="`translate(0 ${sumChartYScale(d)})`"
                    >
                        <line stroke="currentColor" :x2="-6"></line>
                        <text fill="currentColor" :x="-7" dy="0.3em">
                            {{ d }}
                        </text>
                    </g>
                </g>
            </g>

            <g
                id="sumXAxis"
                :transform="`translate(${0} ${
                    sumChartHeight - sumChartMarginBottom - xAxisHeight
                })`"
            >
                <g
                    id="sumXAxisTicks"
                    :font-size="0.6 * xAxisHeight"
                    text-anchor="middle"
                >
                    <g
                        v-for="d in xTicks"
                        :key="d"
                        :transform="`translate(${zoomedXScale(d)},0)`"
                    >
                        <line
                            stroke="currentColor"
                            :y2="0.4 * xAxisHeight"
                        ></line>
                        <text fill="currentColor" :y="xAxisHeight">
                            {{ d }}
                        </text>
                    </g>
                </g>
            </g>

            <g id="sumBars" :clip-path="`url(#sumClip-${props.dbId})`">
                <rect
                    v-for="(d, i) in sumFeatures"
                    :key="i"
                    stroke="none"
                    fill="black"
                    :x="zoomedXScale(i)"
                    :y="sumChartYScale(d)"
                    :width="barWidth * transformRef.k"
                    :height="sumChartYScale(0) - sumChartYScale(d)"
                ></rect>
            </g>
            <clipPath :id="`sumClip-${props.dbId}`">
                <rect
                    :x="sumChartMarginLeft + sumChartYAxisWidth"
                    :y="sumChartMarginTop"
                    :width="barsAreaWidth"
                    :height="
                        sumChartHeight -
                        sumChartMarginTop -
                        sumChartMarginBottom
                    "
                />
            </clipPath>
            <rect
                class="zoomArea"
                :x="sumChartMarginLeft + sumChartYAxisWidth"
                :y="sumChartMarginTop"
                :width="barsAreaWidth"
                :height="
                    sumChartHeight - sumChartMarginTop - sumChartMarginBottom
                "
                stroke="none"
                fill-opacity="0"
                v-zoom
            ></rect>
        </svg>

        <div
            v-if="
                !isDbWiseComparativeDb(db) &&
                isEmptyDict(db.nodesSelections['public'])
            "
            :style="{
                flex: '1 0 0',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
            }"
        >
            <p :style="{ textAlign: 'center' }">
                select nodes for further exploration
            </p>
        </div>

        <template v-else>
            <div
                v-for="(
                    [chartName, chartFeatData, chartColorRange], chartIndex
                ) in isDbWiseComparativeDb(db)
                    ? [
                          [
                              'abs feature diff between nodes from db0 and nodes from db1',
                              dbWiseDiffFeatures, view.diffColorRange
                          ],
                          ['node features from db0', featuresFromDb0 ,view.sel0ColorRange , ],
                          ['node features from db1', featuresFromDb1,view.sel1ColorRange ],
                      ] as [string, number[],[number,number]][]
                    : [
                          [ 'feature diff between all nodes and selected nodes(remaining nodes features)', diffFeatures, view.diffColorRange ],
                          ['selected nodes features', selFeatures,    view.selColorRange ],
                      ] as [string, number[],[number,number]][]"
                :key="chartName"
                class="stripChartBox"
                :style="{
                    flex: '1 0 0',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'center',
                }"
            >
                <div :style="{ textAlign: 'center' }">
                    {{ chartName }}
                </div>
                <svg
                    class="diffSvg"
                    :width="sumChartWidth"
                    :height="
                        stripChartMarginTop + stripChartRectHeight + xAxisHeight
                    "
                >
                    <g
                        class="xAxis"
                        :transform="`translate(${0} ${
                            stripChartMarginTop + stripChartRectHeight
                        })`"
                    >
                        <g
                            id="sumXAxisTicks"
                            :font-size="0.6 * xAxisHeight"
                            text-anchor="middle"
                        >
                            <g
                                v-for="d in xTicks"
                                :key="d"
                                :transform="`translate(${zoomedXScale(d)},0)`"
                            >
                                <line
                                    stroke="currentColor"
                                    :y2="0.4 * xAxisHeight"
                                ></line>
                                <text fill="currentColor" :y="xAxisHeight">
                                    {{ d }}
                                </text>
                            </g>
                        </g>
                    </g>
                    <g
                        class="rectFill"
                        :clip-path="`url(#sparse-feat-clip-${props.dbId}-${chartIndex})`"
                    >
                        <template v-for="(d, i) in chartFeatData" :key="i">
                            <rect
                                v-if="d != 0"
                                :x="zoomedXScale(i)"
                                :y="stripChartMarginTop"
                                :height="stripChartRectHeight"
                                :width="barWidth * transformRef.k"
                                stroke="none"
                                :fill="
                                    d3.interpolate(
                                        d3.interpolateGreys(chartColorRange[0]),
                                        d3.interpolateGreys(chartColorRange[1]),
                                    )(
                                        d / (d3.max(sumFeatures) as number - 0)
                                    )
                                "
                            ></rect>
                        </template>
                    </g>
                    <clipPath
                        :id="`sparse-feat-clip-${props.dbId}-${chartIndex}`"
                    >
                        <rect
                            :x="sumChartMarginLeft + sumChartYAxisWidth"
                            :y="stripChartMarginTop"
                            :width="barsAreaWidth"
                            :height="stripChartRectHeight"
                        />
                    </clipPath>
                    <rect
                        class="rectBorder zoomArea"
                        stroke="black"
                        stroke-width="1"
                        fill-opacity="0"
                        :x="sumChartMarginLeft + sumChartYAxisWidth"
                        :y="stripChartMarginTop"
                        :width="barsAreaWidth"
                        :height="stripChartRectHeight"
                        v-zoom
                    ></rect>
                </svg>
            </div>
        </template>
    </el-scrollbar>
</template>

<script setup lang="ts">
import { useMyStore } from "@/stores/store";
import * as d3 from "d3";
import type {
    CompDashboard,
    SingleDashboard,
    SparseView,
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
    nextTick,
} from "vue";
import { useResize } from "../plugin/useResize";
import { isEmptyDict } from "@/utils/graphUtils";
import type { ScrollbarInstance } from "element-plus";
import { isDbWiseComparativeDb } from "@/types/typeFuncs";
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

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION data prepare
const myStore = useMyStore();
const db = myStore.getTypeReducedDashboardById(props.dbId)!;
const ds = Object.hasOwn(db, "refDatasetsNames")
    ? myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[0]) ||
      myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[1])
    : myStore.getDatasetByName((db as SingleDashboard).refDatasetName);
const view = myStore.getViewByName(db, props.viewName) as SparseView;

// console.log(
//     "in sparse , ds.feature...",
//     ds.nodeSparseFeatureIndexToSemantics,
//     ds.nodeSparseFeatureIndexes,
//     ds.nodeSparseFeatureValues,
//     ds.numNodeSparseFeatureDims
// );
const predLabels = ds.predLabels || [];
const trueLabels = ds.trueLabels || [];
const colorScale = ds.colorScale || (() => "#888");

// const srcNodesDict = computed(() => db.nodesSelections["public"]);
const srcEntryIds = myStore.getViewSourceNodesSelectionEntry(props.viewName);
const tgtEntryIds = myStore.getViewTargetNodesSelectionEntry(props.viewName);
const curSrcEntryId = ref<Type_NodesSelectionEntryId>(srcEntryIds[0]);

const { numNodeSparseFeatureDims } = ds;
////////////////// !SECTION data prepare
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION sum chart
const sumFeatures = ds.nodeSparseFeatureValues.reduce((acc, cur, curI) => {
    const curLongFeature = Array.from({
        length: numNodeSparseFeatureDims,
    }).fill(0) as number[];
    cur.forEach((d, i) => {
        curLongFeature[ds.nodeSparseFeatureIndexes[curI][i]] = d;
    });
    return acc.map((feat, i) => feat + curLongFeature[i]);
}, Array.from({ length: numNodeSparseFeatureDims }).fill(0) as number[]);

const sumChartHeight = computed(() => view.bodyHeight * 0.3);
const sumChartWidth = computed(() => view.bodyWidth);
const sumChartMarginLeft = computed(() => sumChartWidth.value * 0.03);
const sumChartMarginRight = computed(() => sumChartWidth.value * 0.01);
const sumChartMarginTop = computed(() => sumChartHeight.value * 0.03);
const sumChartMarginBottom = computed(() => sumChartHeight.value * 0.01);
const xAxisHeight = computed(() => sumChartHeight.value * 0.12);
const sumChartYAxisWidth = computed(() => sumChartWidth.value * 0.02);
const barsAreaWidth = computed(
    () =>
        sumChartWidth.value -
        sumChartMarginLeft.value -
        sumChartMarginRight.value -
        sumChartYAxisWidth.value
);
const barWidth = computed(() => barsAreaWidth.value / numNodeSparseFeatureDims);
const xScale = computed(() =>
    d3
        .scaleLinear()
        .domain([0, numNodeSparseFeatureDims])
        .range([
            sumChartMarginLeft.value + sumChartYAxisWidth.value,
            sumChartWidth.value - sumChartMarginRight.value,
        ])
);
const sumChartYScale = computed(() =>
    d3
        .scaleLinear()
        .domain([0, d3.max(sumFeatures)] as [number, number])
        .nice()
        .range([
            sumChartHeight.value -
                sumChartMarginBottom.value -
                xAxisHeight.value,
            sumChartMarginTop.value,
        ])
);
const xTickNum = 12;
const sumChartYTickNum = 8;
// const xTicks = computed(() => xScale.value.ticks(xTickNum));
const sumChartYTicks = computed(() =>
    sumChartYScale.value.ticks(sumChartYTickNum)
);
////////////////// !SECTION sum chart
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION diff & sel
const accFeatByNodesDict = (
    nodesDict: Record<Type_NodeId, any>,
    numDims: number
) => {
    return ds.nodeSparseFeatureValues.reduce((acc, cur, curI) => {
        if (nodesDict[curI + ""]) {
            const curLongFeature = Array.from({ length: numDims }).fill(
                0
            ) as number[];
            cur.forEach((d, i) => {
                curLongFeature[ds.nodeSparseFeatureIndexes[curI][i]] = d;
            });
            return acc.map((feat, i) => feat + curLongFeature[i]);
        } else {
            return acc;
        }
    }, Array.from({ length: numDims }).fill(0) as number[]);
};
const srcNodesDict = computed(() =>
    isDbWiseComparativeDb(db)
        ? db.nodesSelections["full"]
        : db.nodesSelections["public"]
);
const selFeatures = computed(() =>
    accFeatByNodesDict(srcNodesDict.value, numNodeSparseFeatureDims)
);
const diffFeatures = computed(() =>
    sumFeatures.map((d, i) => d - selFeatures.value[i])
);

const stripChartRectHeight = computed(() => view.bodyHeight * 0.05);
const stripChartMarginTop = computed(() => view.bodyHeight * 0.01);

////////////////// !SECTION diff & sel
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION dbWiseComp
const nodesDictFromDb0 = computed(() => {
    const ret: Record<
        string,
        { gid: string; parentDbIndex: number; hop: number }
    > = {};
    if (!isDbWiseComparativeDb(db)) {
        return ret;
    }
    for (const id in srcNodesDict.value) {
        const { parentDbIndex } = srcNodesDict.value[id] as {
            gid: string;
            parentDbIndex: number;
            hop: number;
        };
        if (parentDbIndex === 0 || parentDbIndex === 2) {
            ret[id] = srcNodesDict.value[id];
        }
    }
    return ret;
});
const nodesDictFromDb1 = computed(() => {
    const ret: Record<
        string,
        { gid: string; parentDbIndex: number; hop: number }
    > = {};
    if (!isDbWiseComparativeDb(db)) {
        return ret;
    }
    for (const id in srcNodesDict.value) {
        const { parentDbIndex } = srcNodesDict.value[id] as {
            gid: string;
            parentDbIndex: number;
            hop: number;
        };
        if (parentDbIndex === 1 || parentDbIndex === 2) {
            ret[id] = srcNodesDict.value[id];
        }
    }
    return ret;
});
const featuresFromDb0 = computed(() =>
    accFeatByNodesDict(nodesDictFromDb0.value, numNodeSparseFeatureDims)
);
const featuresFromDb1 = computed(() =>
    accFeatByNodesDict(nodesDictFromDb1.value, numNodeSparseFeatureDims)
);
const dbWiseDiffFeatures = computed(() =>
    featuresFromDb0.value.map((d, i) => Math.abs(d - featuresFromDb1.value[i]))
);
////////////////// !SECTION dbWiseComp
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION zoom
const transformRef = ref(d3.zoomIdentity);

const zoom = d3
    .zoom<SVGElement, unknown>()
    .extent([
        [0, 0],
        [
            barsAreaWidth.value,
            sumChartHeight.value -
                sumChartMarginTop.value -
                sumChartMarginBottom.value,
        ],
    ])
    .translateExtent([
        [0, 0],
        [
            barsAreaWidth.value,
            sumChartHeight.value -
                sumChartMarginTop.value -
                sumChartMarginBottom.value,
        ],
    ])
    .scaleExtent([1, 128])
    .on("zoom", function (event: d3.D3ZoomEvent<SVGElement, void>) {
        // console.log("in sparseView, on zoom, event", event);
        // console.log("in sparseView, on zoom, this", this);
        // console.log(
        //     "in sparseView, on zoom, extract the transformData of this, d3.zoomTransform(this)",
        //     d3.zoomTransform(this)
        // );

        transformRef.value = event.transform;
        if (event.sourceEvent) {
            //这个判断，可以避免在resetZoom时调用，因为这个动作在这些rect元素上没有真正的事件

            //以下代码是为了：一个strip放大，其他的也同步
            const rects = d3.selectAll(".zoomArea");
            // console.log( "all rects' transform is:", rects.nodes().map((d) => d3.zoomTransform(d)));
            // console.log( "all rects' __zoom is:", rects.nodes().map((d) => d.__zoom));
            rects.nodes().forEach((d, i, arr) => {
                arr[i]!.__zoom = event.transform; //NOTE - bad code modifying private attrs directly!
            });
        }
    });
const zoomedXScale = computed(() =>
    transformRef.value.rescaleX(xScale.value).interpolate(d3.interpolateRound)
);
const xTicks = computed(() => zoomedXScale.value.ticks(xTickNum));
const vZoom = {
    mounted(el: SVGElement) {
        console.log("in sparse vZoom, mounted!", d3.select(el).node());
        d3.select(el).call(zoom, d3.zoomIdentity);

        view.resetZoomFunc = () => {
            d3.select(".zoomArea").transition().duration(750).call(
                //我们仅需第一个即可，后面rect都是根据 transformRef 改的

                zoom.transform, //这个函数会调用onZoom
                d3.zoomIdentity
                // d3
                //     .zoomTransform(d3.zoomIdentity)
                //     .invertX(heatStripWidth.value / 2)
            );
        };
    },
    unmounted() {
        zoom.on("zoom", null);

        view.resetZoomFunc = () => {};
    },
};
watch(
    [() => view.bodyWidth, () => view.bodyHeight],
    () => {
        zoom.extent([
            [0, 0],
            [
                barsAreaWidth.value,
                sumChartHeight.value -
                    sumChartMarginTop.value -
                    sumChartMarginBottom.value,
            ],
        ]).translateExtent([
            [0, 0],
            [
                barsAreaWidth.value,
                sumChartHeight.value -
                    sumChartMarginTop.value -
                    sumChartMarginBottom.value,
            ],
        ]);
    },
    { immediate: true, flush: "post" }
);
////////////////// !SECTION zoom
////////////////////////////////////////////////////////////////////////////////
</script>

<style scoped></style>
