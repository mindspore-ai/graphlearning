<template>
    <el-scrollbar
        :height="view.bodyHeight"
        :style="{ width: `${view.bodyWidth}px` }"
    >
        <div
            class="grid-multi-histogram"
            :style="{
                gridTemplateColumns: `repeat(${view.numColumns}, 1fr)`,
                gridTemplateRows: `repeat(${numRows}, 1fr)`,
                'box-sizing': 'border-box',
            }"
            ref="gridRef"
        >
            <div
                class="block"
                v-for="(featureName, featureIndex) in featureNames"
                :key="featureName"
                ref="subViewRefs"
            >
                <svg
                    :height="subHeight"
                    :width="subWidth"
                    ref="svgRefs"
                    version="1.1"
                    xmlns="http://www.w3.org/2000/svg"
                    baseProfile="full"
                >
                    <g
                        class="margin"
                        :transform="`translate(${mL} ${mT})`"
                        :height="subHeight - mT - mB"
                        :width="subWidth - mL - mR"
                    >
                        <g
                            class="title"
                            :height="titleHeight * 0.8"
                            :width="subWidth - mL - mR"
                            :transform="`translate(${0} 0)`"
                        >
                            <text
                                class="hiddenTitle"
                                ref="hiddenTitle"
                                fill="none"
                                text-anchor="middle"
                                :x="(subWidth - mL - mR) / 2"
                                :font-size="hiddenTitleFontSize"
                                :dy="titleHeight * 0.2"
                            >
                                {{ featureName }}
                            </text>
                            <text
                                cursor="grab"
                                class="realTitle"
                                fill="black"
                                text-anchor="middle"
                                :x="(subWidth - mL - mR) / 2"
                                :font-size="realTitleFontSize"
                                :dy="titleHeight * 0.2"
                            >
                                <title>drag to swap</title>
                                {{ featureName }}
                            </text>
                        </g>

                        <g
                            class="yAxisName"
                            :transform="`translate(${subWidth * 0.01} ${
                                titleHeight * 0.8
                            })`"
                            :font-size="realTitleFontSize * 0.8"
                        >
                            <text>
                                {{
                                    view.isRelative
                                        ? "relative frequency"
                                        : "frequency"
                                }}
                            </text>
                        </g>
                        <g
                            class="yAxis"
                            :height="
                                subHeight - mT - mB - titleHeight - xAxisHeight
                            "
                            :width="yAxisWidth"
                            :transform="`translate(${yAxisWidth} ${titleHeight})`"
                        >
                            <g class="yAxisTicks" text-anchor="end">
                                <g
                                    v-for="d in yTicksArr[featureIndex]"
                                    :key="d"
                                    :transform="`translate(0 ${yScaleArr[
                                        featureIndex
                                    ](d)})`"
                                >
                                    <line stroke="currentColor" :x2="-6"></line>
                                    <text
                                        :font-size="
                                            ((subHeight -
                                                mT -
                                                mB -
                                                xAxisHeight -
                                                titleHeight) /
                                                yTicksNum) *
                                            0.6
                                        "
                                        fill="currentColor"
                                        :x="-7"
                                        dy="0.3em"
                                    >
                                        {{ d }}
                                    </text>
                                </g>
                            </g>
                        </g>

                        <g
                            class="xAxisName"
                            :transform="`translate(${subWidth * 0.95} ${
                                subHeight * 0.95
                            })`"
                            :font-size="realTitleFontSize * 0.8"
                        >
                            <text text-anchor="end">feature buckets</text>
                        </g>
                        <g
                            class="xAxis"
                            :height="xAxisHeight"
                            :width="subWidth - mL - mR - yAxisWidth"
                            :transform="`translate(${yAxisWidth} ${
                                subHeight - mB - mT - xAxisHeight
                            })`"
                        >
                            <g class="xAxisTicks" text-anchor="start">
                                <g
                                    v-for="d in xTicksArr[featureIndex]"
                                    :key="d"
                                    :transform="`translate(${xScaleArr[
                                        featureIndex
                                    ](d)},0)`"
                                >
                                    <line stroke="currentColor" :y2="6"></line>
                                    <text
                                        :font-size="
                                            ((subHeight -
                                                mT -
                                                mB -
                                                xAxisHeight -
                                                titleHeight) /
                                                yTicksNum) *
                                            0.6
                                        "
                                        fill="currentColor"
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
                            class="bars"
                            :transform="`translate(${yAxisWidth} ${titleHeight})`"
                        >
                            <rect
                                class="barsBorder"
                                x="0"
                                y="0"
                                :width="subWidth - mL - mR - yAxisWidth"
                                :height="
                                    subHeight -
                                    mT -
                                    mB -
                                    titleHeight -
                                    xAxisHeight
                                "
                                stroke="black"
                                stroke-width="1"
                                fill-opacity="0"
                                @dblclick="
                                    (e) => {
                                        if (db.clearSelMode === 'auto') {
                                            tgtEntryIds.forEach((entryId) => {
                                                db.nodesSelections[entryId] =
                                                    {};
                                            });
                                        }
                                        e.stopPropagation();
                                    }
                                "
                            >
                                <title>
                                    double click to cancel selection(only in
                                    auto clear mode)
                                </title>
                            </rect>
                            <rect
                                class="bar"
                                v-for="(d, i) in buckets[featureIndex]"
                                :key="i"
                                stroke="black"
                                stroke-width="1"
                                fill="none"
                                :x="xScaleArr[featureIndex](d.x0!) + xInsetLeft"
                                :width="
                        Math.max(
                            0,
                            xScaleArr[featureIndex](d.x1!) -
                                xScaleArr[featureIndex](d.x0!) -
                                xInsetLeft -
                                xInsetRight
                        )
                    "
                                :y="yScaleArr[featureIndex](d.length)"
                                :height="
                                    Math.max(
                                        0,
                                        yScaleArr[featureIndex](0) -
                                            yScaleArr[featureIndex](d.length)
                                    )
                                "
                            ></rect>
                            <g
                                class="stackedBars"
                                v-for="(d, i) in buckets[featureIndex]"
                                :key="i"
                            >
                                <!-- entry [string,number]表示当前buck中某个label有多少个-->
                                <TransitionGroup
                                    tag="g"
                                    name="stack"
                                    :css="false"
                                >
                                    <rect
                                        v-for="entry in sortedAccumulatedCount[
                                            featureIndex
                                        ][i]"
                                        :key="entry.key"
                                        :y="
                                            yScaleArr[featureIndex](
                                                entry.cumulativeCount
                                            )
                                        "
                                        :height="
                                            Math.max(
                                                0,
                                                yScaleArr[featureIndex](0) -
                                                    yScaleArr[featureIndex](
                                                        entry.count
                                                    )
                                            )
                                        "
                                        :x="xScaleArr[featureIndex](d.x0!) + xInsetLeft"
                                        :width="Math.max( 0, xScaleArr[featureIndex](d.x1!) - xScaleArr[featureIndex](d.x0!) - xInsetLeft - xInsetRight)"
                                        stroke="none"
                                        :fill="colorScale(entry.key)"
                                        cursor="pointer"
                                        @mouseenter="
                                            db.highlightedNodeIds =
                                                entry.nodeIds.reduce(
                                                    (acc, cur) => ({
                                                        ...acc,
                                                        [cur]: true,
                                                    }),
                                                    {}
                                                )
                                        "
                                        @mouseleave="db.highlightedNodeIds = {}"
                                        @click="
                                            async (e) => {
                                                curPriorSubViewIndex =
                                                    featureIndex;
                                                curPriorLabel = entry.key;

                                                await nextTick();
                                                e.stopPropagation();
                                            }
                                        "
                                        @dblclick="
                                            (e) => {
                                                if (
                                                    db.clearSelMode === 'auto'
                                                ) {
                                                    tgtEntryIds.forEach(
                                                        (entryId) => {
                                                            db.nodesSelections[
                                                                entryId
                                                            ] = {};
                                                        }
                                                    );
                                                }
                                                db.fromViewName =
                                                    props.viewName;
                                                tgtEntryIds.forEach(
                                                    (entryId) => {
                                                        db.nodesSelections[
                                                            entryId
                                                        ] = {
                                                            ...entry.nodeIds.reduce(
                                                                (acc, cur) => ({
                                                                    ...acc,
                                                                    [cur]: true,
                                                                }),
                                                                {}
                                                            ),
                                                            ...db
                                                                .nodesSelections[
                                                                entryId
                                                            ],
                                                        };
                                                    }
                                                );
                                                e.stopPropagation();
                                            }
                                        "
                                    >
                                        <title>
                                            {{
                                                `label: ${entry.key}, count: ${entry.count}\nclick to stack to bottom\ndouble click to select`
                                            }}
                                        </title>
                                    </rect>
                                </TransitionGroup>
                            </g>
                        </g>
                        <g
                            class="barValues"
                            font-size="0.8em"
                            text-anchor="middle"
                            :transform="`translate(${yAxisWidth} ${titleHeight})`"
                        >
                            <text
                                class="barValue"
                                v-for="(d, i) in buckets[featureIndex]"
                                :key="i"
                                :x="d.length>0?((xScaleArr[featureIndex](d.x0!) + xScaleArr[featureIndex](d.x1!)) / 2) : 0"
                                :y="
                                    d.length > 0
                                        ? d.length >
                                          (yScaleArr[featureIndex].domain()[1] -
                                              yScaleArr[
                                                  featureIndex
                                              ].domain()[0]) *
                                              0.7
                                            ? yScaleArr[featureIndex](
                                                  d.length
                                              ) -
                                              svgEmFontSize * 0.8
                                            : yScaleArr[featureIndex](
                                                  d.length
                                              ) -
                                              svgEmFontSize * 0.8
                                        : 0
                                "
                                dy="0.3em"
                                cursor="pointer"
                                @mouseenter="
                                    db.highlightedNodeIds = d.reduce(
                                        (acc, cur) => ({
                                            ...acc,
                                            [cur.id]: true,
                                        }),
                                        {}
                                    )
                                "
                                @mouseleave="db.highlightedNodeIds = {}"
                                @dblclick="
                                    (e) => {
                                        if (db.clearSelMode === 'auto') {
                                            tgtEntryIds.forEach((entryId) => {
                                                db.nodesSelections[entryId] =
                                                    {};
                                            });
                                        }

                                        tgtEntryIds.forEach((entryId) => {
                                            db.nodesSelections[entryId] = {
                                                ...d.reduce(
                                                    (acc, cur) => ({
                                                        ...acc,
                                                        [cur.id]: true,
                                                    }),
                                                    {}
                                                ),
                                                ...db.nodesSelections[entryId],
                                            };
                                        });
                                        e.stopPropagation();
                                    }
                                "
                            >
                                <!-- 上面的y属性，可以动态控制text在bar上面还是下面（因为有时候放不下）-->
                                {{ d.length > 0 ? d.length : "" }}
                                <title>double click to select this bar</title>
                            </text>
                        </g>
                    </g>
                </svg>
            </div>
        </div>
    </el-scrollbar>
</template>

<script setup lang="ts">
/** @description
 * This view is for dense node features
 */
import { useMyStore } from "@/stores/store";
import * as d3 from "d3";
import type {
    CompDashboard,
    DenseView,
    NodeCoord,
    SingleDashboard,
    TsneCoord,
    Type_NodeId,
    Type_NodesSelectionEntryId,
} from "@/types/types";
import {
    computed,
    nextTick,
    watch,
    ref,
    onMounted,
    onUnmounted,
    onBeforeUnmount,
} from "vue";
import { useResize } from "../plugin/useResize";
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

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION data prepare
const myStore = useMyStore();
const db = myStore.getTypeReducedDashboardById(props.dbId)!;
const ds = Object.hasOwn(db, "refDatasetsNames")
    ? myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[0]) ||
      myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[1])
    : myStore.getDatasetByName((db as SingleDashboard).refDatasetName);
const view = myStore.getViewByName(db, props.viewName) as DenseView;
const svgRefs = ref<SVGElement[] | null>(null);

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION selection related
const srcEntryIds = myStore.getViewSourceNodesSelectionEntry(props.viewName);
const tgtEntryIds = myStore.getViewTargetNodesSelectionEntry(props.viewName);

const curSrcEntryId = ref<Type_NodesSelectionEntryId>(srcEntryIds[0]);
const srcNodesDict = computed(() => db.nodesSelections[curSrcEntryId.value]);
const nodesDictFromDb0 = computed(() => {
    const ret = {} as Record<
        Type_NodeId,
        { gid: string; parentDbIndex: number; hop: number }
    >;
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
    const ret = {} as Record<
        Type_NodeId,
        { gid: string; parentDbIndex: number; hop: number }
    >;
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
////////////////// !SECTION selection related
////////////////////////////////////////////////////////////////////////////////

//NOTE 对于 dbWiseComp ，只需从源头改变 featureNames , featureGroupByNames
/**
 * ```
 * [
 *      { id:string, name1:number, name2:number, ...},
 *      { id:string, name1:number, name2:number, ...},
 *      ...,
 *      columns: ['name1','name2', ...]
 * ]
 * ```
 */
const localFeatures = computed(() =>
    ds.denseNodeFeatures.filter((d) => srcNodesDict.value[d.id])
);
const db0Features = computed(() =>
    ds.denseNodeFeatures.filter((d) => nodesDictFromDb0.value[d.id])
);
const db1Features = computed(() =>
    ds.denseNodeFeatures.filter((d) => nodesDictFromDb1.value[d.id])
);
const originFeatureNames = computed(() => ds.denseNodeFeatures.columns);
const featureNames = computed(() =>
    isDbWiseComparativeDb(db)
        ? ds.denseNodeFeatures.columns.flatMap((d) => [d + "_db0", d + "_db1"])
        : ds.denseNodeFeatures.columns
);

/**
 * ```{id:string, feature:number}[][]```\
 * size: len(featureNames) * len(localNodes)
 */
const featureGroupByNames = computed(() =>
    isDbWiseComparativeDb(db)
        ? originFeatureNames.value.flatMap(
              (
                  name //NOTE - 顺序对应
              ) => [
                  db0Features.value.map((d) => ({
                      id: d.id,
                      feature: d[name] as number, //unknown to number
                  })),
                  db1Features.value.map((d) => ({
                      id: d.id,
                      feature: d[name] as number, //unknown to number
                  })),
              ]
          )
        : originFeatureNames.value.map((name) =>
              localFeatures.value.map((d) => ({
                  id: d.id,
                  feature: d[name] as number, //unknown to number
              }))
          )
);

// const db0FeatureGroupByNames = computed(() =>
//     featureNames.value.map((name) =>
//         db0Features.value.map((d) => ({
//             id: d.id,
//             feature: d[name] as number, //unknown to number
//         }))
//     )
// );
// const db1FeatureGroupByNames = computed(() =>
//     featureNames.value.map((name) =>
//         db1Features.value.map((d) => ({
//             id: d.id,
//             feature: d[name] as number, //unknown to number
//         }))
//     )
// );

const binFn = computed(
    () =>
        d3
            .bin<{ id: Type_NodeId; feature: number }, number>()
            .domain(
                (data: Iterable<number>) =>
                    d3
                        .scaleLinear()
                        .domain(d3.extent(data) as [number, number])
                        .nice()
                        .domain() as [number, number]
            )
            .value((d) => d.feature) //NOTE 这个.value跟vue那个.value别晕了
);

/**
 * ```
 * [
 *     [ [{id,feat},{id,feat},{id,feat}], [], [], ...],  //hist arr
 *     [ [], [], [], [], [], ...],                       //hist arr
 *     ...
 * ] //size: len(featureNames) * unknown
 * ```
 */
const buckets = computed(() =>
    featureGroupByNames.value.map((d) => binFn.value(d))
);
// const db0Buckets = computed(() =>
//     db0FeatureGroupByNames.value.map((d) => binFn.value(d))
// );
// const db1Buckets = computed(() =>
//     db1FeatureGroupByNames.value.map((d) => binFn.value(d))
// );

const colorScale = ds.colorScale || (() => "#888");
const trueLabels = ds.trueLabels || [];
const predLabels = ds.predLabels || [];
const currentLabels = computed(() =>
    db.labelType === "true" ? trueLabels : predLabels
);

const countByLabel = computed(() =>
    buckets.value.map(
        (allFeatArr) =>
            allFeatArr.map((arr) =>
                arr.reduce((acc, cur) => {
                    const value = currentLabels.value[+cur.id];
                    // acc[value] = (acc[value] || 0) + 1;
                    acc[value] = [...(acc[value] || []), cur.id];
                    return acc;
                }, {} as Record<number, Type_NodeId[]>)
            ) //NOTE 虽然label是number，但是Object.entries()变成了string
    )
);
const prioritizedCompare =
    (topPriorLabel: string) =>
    (a: [string, ...any[]], b: [string, ...any[]]) => {
        if (a[0] === topPriorLabel && b[0] === topPriorLabel) {
            return 0;
        } else if (a[0] === topPriorLabel) {
            return -1;
        } else if (b[0] === topPriorLabel) {
            return 1;
        } else {
            return a[0].localeCompare(b[0]);
        }
    };
const curPriorLabel = ref("0");
const curPriorSubViewIndex = ref(0);
const sortedAccumulatedCount = computed(() =>
    countByLabel.value.map((allFeatArr, i) =>
        allFeatArr.map((d) => {
            //NOTE remember it's a 2d-arr
            const ret: Array<{
                key: string; //即label
                nodeIds: Type_NodeId[];
                count: number;
                cumulativeCount: number;
            }> = [];

            let cumulativeCount = 0;
            //仅correspondAlign状态下 或者 当前subView sort
            //NOTE - 使用computed将所有subView的数据统一计算，产生后果是：
            // 非correspondAlign状态下，除了点击的那个subView，其他的永远保持最初的label顺序
            if (view.isCorrespondAlign || i == curPriorSubViewIndex.value) {
                Object.entries(d)
                    .sort(prioritizedCompare(curPriorLabel.value))
                    .forEach(([key, nodeIds]) => {
                        cumulativeCount += nodeIds.length; //一定是非空
                        ret.push({
                            key,
                            nodeIds,
                            count: nodeIds.length,
                            cumulativeCount,
                        });
                    });
            } else {
                Object.entries(d).forEach(([key, nodeIds]) => {
                    cumulativeCount += nodeIds.length; //一定是非空
                    ret.push({
                        key,
                        nodeIds,
                        count: nodeIds.length,
                        cumulativeCount,
                    });
                });
            }
            return ret;
        })
    )
);

const maxBins = computed(() =>
    isDbWiseComparativeDb(db)
        ? buckets.value.map(
              (bucketsForOneFeat, i, bucketsArr) =>
                  d3.max(
                      [
                          ...bucketsForOneFeat,
                          ...(i & 1 ? bucketsArr[i - 1] : bucketsArr[i + 1]),
                      ],
                      (d) => d.length
                  ) || 0
          )
        : buckets.value.map(
              (bucketsForOneFeat) =>
                  d3.max(bucketsForOneFeat, (d) => d.length) || 0
          )
);

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION sizes & scales
const numRows = computed(() =>
    Math.ceil(featureNames.value.length / view.numColumns)
);
const gridRef = ref<HTMLDivElement | null>(null);
const subViewRefs = ref<Array<HTMLDivElement> | null>(null);
const subHeight = computed(() => view.subHeight);
// const subWidth = ref(
//     (view.bodyWidth * (1 - 0.01 * (1 + view.numColumns))) / view.numColumns
// );
// watch(
//     () => view.bodyWidth,
//     () => {
//         subWidth.value =
//             subViewRefs.value?.at(0)?.clientWidth ||
//             (view.bodyWidth * (1 - 0.01 * (1 + view.numColumns))) /
//                 view.numColumns;
//     },
//     { immediate: true }
// );
// watch(
//     // () => subViewRefs.value?.at(0)?.clientWidth,
//     () => gridRef.value?.firstElementChild?.clientWidth,
//     (newV) => {
//         if (newV) {
//             console.log(
//                 "in dense, subView's clientWidth changed to ",
//                 // newV[0].clientWidth
//                 newV
//             );
//             // subWidth.value = newV[0].clientWidth;
//             subWidth.value = newV;
//         }
//     },
//     { immediate: true, deep: true }
// );
const subWidth = computed(
    () =>
        (view.bodyWidth * (1 - 0.02 * (1 + view.numColumns))) / view.numColumns
    // subViewRefs.value ? subViewRefs.value[0].clientWidth : 0
    // gridRef.value?.firstElementChild?.clientWidth || 0
    // svgRefs.value ? svgRefs.value[0].clientWidth : 0
    // document.getElementById("grid")?.firstElementChild?.clientWidth || 0
);

const titleHeight = computed(() => 0.08 * subHeight.value); //NOTE 让title和yAxisName共用这一部分
const yAxisWidth = computed(() => 0.06 * subWidth.value);
const xAxisHeight = computed(() => 0.15 * subHeight.value);
// NOTE we treat view.bodyMargins as the margins of each sub view
const mL = computed(() => subWidth.value * view.bodyMargins.left);
const mR = computed(() => subWidth.value * view.bodyMargins.right);
const mT = computed(() => subHeight.value * view.bodyMargins.top);
const mB = computed(() => subHeight.value * view.bodyMargins.bottom);
const xInsetLeft = 1;
const xInsetRight = 1;
const xScaleArr = computed(() =>
    buckets.value.map((bucketsForOneFeat, i, bucketsArr) =>
        d3
            .scaleLinear()
            .domain(
                isDbWiseComparativeDb(db)
                    ? [
                          i & 1 //奇偶
                              ? Math.min(
                                    bucketsArr[i - 1][0].x0!,
                                    bucketsArr[i][0].x0!
                                )
                              : Math.min(
                                    bucketsArr[i + 1][0].x0!,
                                    bucketsArr[i][0].x0!
                                ),

                          i & 1
                              ? Math.max(
                                    bucketsArr[i - 1][
                                        bucketsArr[i - 1].length - 1
                                    ].x1!,
                                    bucketsArr[i][bucketsArr[i].length - 1].x1!
                                )
                              : Math.max(
                                    bucketsArr[i + 1][
                                        bucketsArr[i + 1].length - 1
                                    ].x1!,
                                    bucketsArr[i][bucketsArr[i].length - 1].x1!
                                ),
                      ]
                    : ([
                          bucketsForOneFeat[0].x0,
                          bucketsForOneFeat[bucketsForOneFeat.length - 1].x1,
                      ] as [number, number])
            )
            .range([
                0, //NOTE inner g, we start from 0
                subWidth.value - mL.value - mR.value - yAxisWidth.value,
            ])
    )
);

const yScaleArr = computed(() =>
    maxBins.value.map((maxBinForOneFeat) =>
        d3
            .scaleLinear()
            .domain([0, maxBinForOneFeat])
            .range([
                subHeight.value -
                    mB.value -
                    mT.value -
                    xAxisHeight.value -
                    titleHeight.value,
                20, //留出最高的上方字体位置
            ])
            .nice()
    )
);
const xTicksNum = 12;
const xTicksArr = computed(() =>
    xScaleArr.value.map((xScale) => xScale.ticks(xTicksNum))
);

const yTicksNum = computed(
    () => (xTicksNum * subHeight.value) / subWidth.value
);
const yTicksArr = computed(() =>
    yScaleArr.value.map((yScale) => yScale.ticks(yTicksNum.value))
);

////////////////// !SECTION sizes & scales
////////////////////////////////////////////////////////////////////////////////

////////////////// !SECTION data prepare
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION adaptive font size
const svgEmFontSize = computed(() =>
    svgRefs.value
        ? Number.parseFloat(getComputedStyle(svgRefs.value[0]).fontSize)
        : 12
);
const hiddenTitleFontSize = 12;
const hiddenTitle = ref<Array<SVGTextElement> | null>(null);
const realTitleFontSize = ref<number>(12);
// watch(
//     // () => hiddenTitle.value?.at(0)?.clientWidth,
//     hiddenTitle,
//     (newV) => {
//         if (newV) {
//             // console.log(
//             //     "in db",
//             //     db.name,
//             //     "in dense, watch () => hiddenTitle.value?at(0).clientWidth, got newV",
//             //     newV
//             // );
//             realTitleFontSize.value =
//                 (hiddenTitleFontSize *
//                     (subWidth.value - mL.value - mR.value) *
//                     0.7) /
//                 // 100;
//             // (newV || 30);
//             (newV[0]?.clientWidth || 100);
//         }
//     },
//     { immediate: true, deep: true }
// );
//NOTE db切换时，其clientWidth变成了0
// 因为computed跟踪的仍然是其指针，属性变了你是取不到的
// const realTitleFontSize = computed(() => {
//     console.log(
//         "in db",
//         db.name,
//         "in dense",
//         "\nhiddenTitle.value",
//         hiddenTitle.value,
//         "\nhiddenTitle.value.clientWidth",
//         hiddenTitle.value?.clientWidth
//     );
//     return hiddenTitle.value && hiddenTitle.value.clientWidth > 0
//         ? (hiddenTitleFontSize * (subWidth.value - mL.value - mR.value) * 0.7) /
//               hiddenTitle.value.clientWidth
//         : hiddenTitleFontSize;
// });
////////////////// !SECTION adaptive font size
////////////////////////////////////////////////////////////////////////////////
</script>

<style scoped>
.grid-multi-histogram {
    display: grid;
    padding: 1%;
    gap: 1%;
    font-size: 12px;
    /* border: 1px solid black; */
}
.block {
    box-sizing: border-box;
    border: 1px solid black;
    margin: 0;
    padding: 0;
}

/* 1. 声明过渡效果 */
.stack-move,
.stack-enter-active,
.stack-leave-active {
    transition: all 0.5s ease;
}

/* 2. 声明进入和离开的状态 */
.stack-enter-from,
.stack-leave-to {
    opacity: 0;
    transform: scaleY(0.01);
}

/* 3. 确保离开的项目被移除出了布局流
      以便正确地计算移动时的动画效果。 */
.stack-leave-active {
    position: absolute;
}
</style>
