<template>
    <div>
        <LoadingComp v-if="isLoading" text="calculating tree"></LoadingComp>
        <template v-else>
            <span
                >select one db to go to, select two dbs to compare
                {{ " " }}
            </span>
            <el-button
                @click="(e) => handleGotoClick(e, selected2Db[0])"
                :disabled="selected2Db.length === 0 || selected2Db.length === 2"
                type="primary"
                >go to
            </el-button>
            <el-button
                :disabled="selected2Db.length === 0"
                @click="handleCancelClick"
                >cancel</el-button
            >
            <el-button
                :disabled="selected2Db.length === 0 || selected2Db.length === 1"
                type="primary"
                @click="handleCompareClick"
                >compare</el-button
            >
            <el-scrollbar :height="300">
                <div :style="{ fontSize: '10px' }">Dashboards</div>
                <svg
                    :width="props.width"
                    :height="centeredHeight + dbWiseCompAreaHeight"
                    style="max-width: 100%; height: auto; height: intrinsic"
                    font-size="20"
                    :viewBox="`${[
                        (-dy * horizontalPaddingStep) / 2,
                        x0 - dx,
                        props.width,
                        centeredHeight + dbWiseCompAreaHeight,
                    ]}`"
                >
                    <g
                        class="comp-links"
                        stroke="red"
                        stroke-opacity="0.4"
                        stroke-width="2"
                        fill="none"
                    >
                        <path
                            v-for="(curve, i) in dbWiseCompCurveStrings"
                            stroke-dasharray=""
                            :key="i"
                            :d="curve"
                        >
                            <title>
                                {{ "which(0 or 1): " + (i & 1 ? "1" : "0") }}
                            </title>
                        </path>
                    </g>
                    <g
                        class="tree-links"
                        fill="none"
                        :stroke="strokeColor"
                        :stroke-opacity="strokeOpacity"
                        :stroke-linecap="strokeLinecap"
                        :stroke-linejoin="strokeLinejoin"
                        :stroke-width="strokeWidth"
                    >
                        <path
                            v-for="(curve, i) in curveStrings"
                            :key="i"
                            :d="curve"
                        ></path>
                    </g>
                    <g class="tree-nodes">
                        <g
                            v-for="d in root!.descendants()"
                            :key="d.id"
                            cursor="pointer"
                            @click="(e) => handleRectClick(d.id!)"
                            :transform="`translate(${d.y},${d.x})`"
                        >
                            <rect
                                :x="-dx / 2 - 1"
                                :y="-dx / 2 - 1"
                                :width="dx + 2"
                                :height="dx + 2"
                                :stroke="selected2Db.includes(d.id!)? '#409EFF' :'black'"
                                :stroke-width="selected2Db.includes(d.id!)? 3 :1"
                                fill="white"
                                :fill-opacity="0"
                            >
                                <title>
                                    {{
                                        `Name: ${
                                            d.data.name
                                        } \nDate: ${new Date(d.data.date)}\n`
                                    }}
                                </title>
                            </rect>
                            <image
                                :xlink:href="d.data.principalView!.snapshotBase64"
                                :x="
                                    d.data.snapshotWidth > d.data.snapshotHeight
                                        ? dx / -2
                                        : (dx * d.data.snapshotWidth) /
                                          d.data.snapshotHeight /
                                          -2
                                "
                                :width="
                                    d.data.snapshotWidth > d.data.snapshotHeight
                                        ? dx
                                        : (dx * d.data.snapshotWidth) /
                                          d.data.snapshotHeight
                                "
                                :height="
                                    d.data.snapshotWidth > d.data.snapshotHeight
                                        ? (dx * d.data.snapshotHeight) /
                                          d.data.snapshotWidth
                                        : dx
                                "
                                :y="
                                    d.data.snapshotWidth > d.data.snapshotHeight
                                        ? (dx * d.data.snapshotHeight) /
                                          d.data.snapshotWidth /
                                          -2
                                        : dx / -2
                                "
                            >
                                <title>
                                    {{
                                        `Name: ${
                                            d.data.name
                                        } \nDate: ${new Date(d.data.date)}\n`
                                    }}
                                </title>
                            </image>
                        </g>
                    </g>
                    <g
                        class="dbWiseCompTitle"
                        :transform="`translate(${
                            (-dy * horizontalPaddingStep) / 2
                        } ${x0 - dx + centeredHeight})`"
                        :width="props.width - 2 * dbWiseCompMargin"
                        :height="dbWiseCompTitleHeight"
                    >
                        <text font-size="10">Dashboard-wise comparison</text>
                    </g>
                    <g
                        class="comp-nodes"
                        :transform="`translate(${
                            (-dy * horizontalPaddingStep) / 2 + dbWiseCompMargin
                        } ${
                            x0 -
                            dx +
                            centeredHeight +
                            dbWiseCompTitleHeight +
                            dbWiseCompMargin
                        })`"
                        :width="props.width - 2 * dbWiseCompMargin"
                        :height="dbWiseCompAreaHeight"
                    >
                        <g
                            v-for="(db, i) in dbWiseCompList"
                            :key="db.id"
                            :transform="`translate(${
                                (i % dbWiseCompColNum) *
                                (dbWiseCompCellWidth + dbWiseCompGapW)
                            } ${
                                Math.floor(i / dbWiseCompColNum) *
                                (dbWiseCompCellHeight + dbWiseCompGapH)
                            })`"
                            @click="(e) => handleGotoClick(undefined, db.id!)"
                        >
                            <rect
                                :width="dbWiseCompCellWidth"
                                :height="dbWiseCompCellHeight"
                                stroke="black"
                                stroke-width="1"
                                fill="white"
                                fill-opacity="0"
                            >
                                <title>
                                    {{
                                        `Name: ${db.name} \nDate: ${new Date(
                                            db.date
                                        )}\n`
                                    }}
                                </title>
                            </rect>
                            <image
                                :xlink:href="
                                    myStore.getPrincipalSnapshotOfDashboard(db)
                                "
                                :x="1"
                                :width="dbWiseCompCellWidth - 2"
                                :height="dbWiseCompCellHeight - 2"
                                :y="1"
                            >
                                <title>
                                    {{
                                        `Name: ${db.name} \nDate: ${new Date(
                                            db.date
                                        )}\n`
                                    }}
                                </title>
                            </image>
                        </g>
                    </g>
                </svg>
            </el-scrollbar>
        </template>
    </div>
</template>

<script setup lang="ts">
import { ref, watch, nextTick, computed } from "vue";
import { useMyStore } from "../../stores/store";
import LoadingComp from "../state/Loading.vue";
import * as d3 from "d3";
import { getImageDimensionsFromBase64Async } from "@/utils/otherUtils";
import type {
    CompDashboard,
    Dashboard,
    RecentDb,
    SingleDashboard,
    Type_GraphId,
    Type_NodeId,
    View,
} from "@/types/types";
import { nanoid } from "nanoid";

const props = defineProps({
    height: {
        type: Number,
        required: false,
        default: 400,
    },
    width: {
        type: Number,
        required: false,
        default: 1000,
    },
    isSingle: {
        type: Boolean,
        default: true,
    },
    //NOTE 将计算过程和calc状态放在哪，决定了父亲组件和子组件的组织形式，也决定了用几个状态量（才能压得住）
});
declare type TreeNodeDatum = {
    id: string;
    date: string | number | Date;
    name: string;
    parentId: string;
    principalView: View | undefined;
    snapshotWidth: number;
    snapshotHeight: number;
};
const myStore = useMyStore();

const isLoading = ref(true);
const mixedDbList = props.isSingle
    ? myStore.singleDashboardList
    : myStore.compDashboardList;

const dbList = computed(() =>
    mixedDbList.filter((d) => typeof d.parentId === "string")
);
const curveGenerator = d3
    .link(d3.curveBumpX)
    .x((d) => d.y)
    .y((d) => d.x);
const horizontalPaddingStep = 1;

const strokeColor = ref("#555");
const strokeWidth = ref(3); // stroke width for links
const strokeOpacity = ref(0.8); // stroke opacity for links
const strokeLinejoin = ref("round"); // stroke line join for links
const strokeLinecap = ref("round"); // stroke line cap for links

const root = ref<d3.HierarchyNode<TreeNodeDatum> | null>(null);
const dx = ref(80); //treat dx as 1 pixel unit
const mapWidthToHeight = (originHeight: number, originWidth: number) => {
    //Treat dy as the width of each node
    (dy.value * originHeight) / originWidth;
};
const dy = ref<number>(1);
const x0 = ref(Infinity);
const x1 = ref(-x0.value);
const centeredHeight = ref(0);
const curveStrings = ref([]);

const calcTree = async () => {
    // return new Promise((resolve, reject) => {
    const snapshotsSizeArr = await Promise.all(
        dbList.value.map(async (d, i) => {
            try {
                const principalView = myStore.getPrincipalViewOfDashboard(d);
                // console.log(
                //     "in calcTree, get each principal view of each db, now get db:",
                //     d.name,
                //     "and its view",
                //     principalView
                // );
                // if (principalView && principalView.snapshotBase64)
                return getImageDimensionsFromBase64Async(
                    principalView!.snapshotBase64
                );
                // else return new Promise<>((resolve, reject) => {

                // });
            } catch (e) {
                console.warn(
                    `getPrincipalViewOfDashboard of db ${d.name} failed`,
                    e
                );
            }
        })
    );
    // console.log("in tree, in calcTree, got snapshotsSizeArr", snapshotsSizeArr);

    root.value = d3
        .stratify<TreeNodeDatum>()
        .id((d) => d.id)
        .parentId((d) => d.parentId)(
        //只取部分即可。
        dbList.value.map((d, i) => ({
            id: d.id,
            date: d.date,
            name: d.name,
            parentId: d.parentId,
            principalView: myStore.getPrincipalViewOfDashboard(d),
            snapshotWidth: snapshotsSizeArr[i].width,
            snapshotHeight: snapshotsSizeArr[i].height,
        }))
    );
    // console.log("in tree, root", root);

    dy.value = Math.min(
        // 80,
        props.width / (root.value.height + 2 * horizontalPaddingStep)
    );
    d3.tree<TreeNodeDatum>().nodeSize([dx.value + 4, dy.value])(
        // .separation((a, b) => {
        //     const magnitude =
        //         (dy.value * a.data.principalView.bodyHeight) /
        //         a.data.principalView.bodyWidth;
        //     return a.parent == b.parent
        //         ? (1 * magnitude) / dx.value
        //         : (2 * magnitude) / dx.value;
        // })
        root.value
    );

    // Center the tree.
    root.value.each((d) => {
        if (d.x > x1.value) x1.value = d.x;
        if (d.x < x0.value) x0.value = d.x;
    });
    centeredHeight.value = x1.value - x0.value + dx.value * 2;

    // console.log( "in tree after called tree, root.links()", root.value.links());
    curveStrings.value = root.value.links().map(curveGenerator);
    // console.log("in tree, after called tree , root", root);

    return false;
    // resolve(false);
    // });
};
watch(
    () => dbList.value.length,
    async () => {
        isLoading.value = true;
        isLoading.value = await calcTree();
    },
    { flush: "post", immediate: true }
    // { deep: false }
);
const currentDashboardId = computed(() =>
    props.isSingle
        ? myStore.recentSingleDashboardList.at(-1)?.id || ""
        : myStore.recentCompDashboardList.at(-1)?.id || ""
);

// two db
const selected2Db = ref<string[]>([]);
const handleRectClick = (id: string) => {
    const i = selected2Db.value.findIndex((d) => d === id);
    // console.log("in tree, handleRectClick, id:", id, "findIndex,", i);
    if (i >= 0) {
        selected2Db.value.splice(i, 1);
        return;
    }
    if (selected2Db.value.length < 2) {
        selected2Db.value.push(id);
    } else {
        selected2Db.value.shift();
        selected2Db.value.push(id);
    }
};
const handleGotoClick = async (e: Event | undefined, id: string) => {
    await myStore.calcPrincipalSnapshotOfDashboard(currentDashboardId.value);
    myStore.toSingleDashboardById(id);
    if (e) myStore.repairButtonFocus(e);
};
const handleCancelClick = (e: Event) => {
    if (selected2Db.value.length > 0) selected2Db.value.shift();
    myStore.repairButtonFocus(e);
};
const handleCompareClick = async (e: Event) => {
    const db0 = myStore.getTypeReducedDashboardById(selected2Db.value[0])!;
    const db1 = myStore.getTypeReducedDashboardById(selected2Db.value[1])!;
    const dsName = Object.hasOwn(db0, "refDatasetsNames")
        ? db0.refDatasetsNames[0]
        : db0.refDatasetName;

    const ds = myStore.getDatasetByName(dsName)!;
    const theOtherDs = Object.hasOwn(db0, "refDatasetsNames")
        ? myStore.getDatasetByName((db0 as CompDashboard).refDatasetsNames[1])
        : ds;

    // const sel0 = {};
    // const sel1 = {};

    // for (const id in db0.srcNodesDict) {
    //     if (db0.srcNodesDict[id].hop === -1) sel0[id] = { hop: -1 };
    // }

    // for (const id in db1.srcNodesDict) {
    //     if (db1.srcNodesDict[id].hop === -1) sel1[id] = { hop: -1 };
    // }
    // const selAndNeighbor0 = calcNeighborDict(
    //     sel0,
    //     ds.hops || theOtherDs.hops!,
    //     ds.neighborMasksByHop || theOtherDs.neighborMasksByHop!
    // );
    // const selAndNeighbor1 = calcNeighborDict(
    //     sel1,
    //     ds.hops || theOtherDs.hops!,
    //     ds.neighborMasksByHop || theOtherDs.neighborMasksByHop!
    // );
    const selAndNeighbor0 = { ...db0.srcNodesDict };
    const selAndNeighbor1 = { ...db1.srcNodesDict };
    // console.log( "in tree, in handleCompare,", "\nselAndNeighbor0", selAndNeighbor0, "\nselAndNeighbor1", selAndNeighbor1);
    for (const id in selAndNeighbor0) {
        selAndNeighbor0[id] = {
            ...selAndNeighbor0[id],
            parentDbIndex: 0,
            gid: (ds.globalNodesDict || theOtherDs.globalNodesDict)[id].gid,
        };
    }
    for (const id in selAndNeighbor1) {
        selAndNeighbor1[id] = {
            ...selAndNeighbor1[id],
            parentDbIndex: 1,
            gid: (ds.globalNodesDict || theOtherDs.globalNodesDict)[id].gid,
        };
    }

    const sharedNodesDict: Record<
        Type_NodeId,
        { gid: Type_GraphId; parentDbIndex: number; hop: number }
    > = {};
    for (const id in selAndNeighbor0) {
        sharedNodesDict[id] = selAndNeighbor1[id]
            ? {
                  //NOTE 两者都有，hop小优先
                  hop: d3.min<number, number>(
                      [selAndNeighbor0[id].hop, selAndNeighbor1[id].hop],
                      (d) => Number(d)
                  )!,
                  parentDbIndex:
                      selAndNeighbor0[id].hop === selAndNeighbor1[id].hop
                          ? 2
                          : d3.minIndex(
                                [
                                    selAndNeighbor0[id].hop,
                                    selAndNeighbor1[id].hop,
                                ],
                                (d) => Number(d)
                            ),
                  gid: (ds.globalNodesDict || theOtherDs.globalNodesDict)[id]
                      .gid,
              }
            : selAndNeighbor0[id];
    }
    for (const id in selAndNeighbor1) {
        if (!sharedNodesDict[id]) {
            sharedNodesDict[id] = selAndNeighbor1[id];
        }
    }

    console.log(
        "in tree, in handleCompare,",
        "\nsharedNodesDict",
        sharedNodesDict
    );
    await myStore.calcPrincipalSnapshotOfDashboard(
        myStore.getTypeReducedDashboardById(currentDashboardId.value)!
    );

    if (Object.hasOwn(db0, "refDatasetsNames")) {
        myStore.addCompDashboard(
            {
                id: nanoid(),
                refDatasetsNames: [...(db0 as CompDashboard).refDatasetsNames],
                date: Date.now ? Date.now() : new Date().getTime(),
                isRoot: false,
                isComplete: true,
                parentId: [db0.id, db1.id],
                fromViewName: db0.viewsDefinitionList[0].viewName,
                graphCoordsRet: undefined,
                srcNodesDict: sharedNodesDict,
            },
            db0.viewsDefinitionList
                .map((d) => d.viewName)
                .filter((d) => !(d.match(/Polar/i) || d.match(/Rank/i)))
            // omit two model-wise comparative views
        );
    } else {
        myStore.addSingleDashboard(
            {
                id: nanoid(),
                refDatasetName: (db0 as SingleDashboard).refDatasetName,
                date: Date.now ? Date.now() : new Date().getTime(),
                isRoot: false,
                isComplete: true,
                parentId: [db0.id, db1.id],
                fromViewName: myStore.defaultSingleViewNames[ds.taskType][0],
                graphCoordsRet: undefined,
                srcNodesDict: sharedNodesDict,
            },
            db0.viewsDefinitionList.map((d) => d.viewName) //NOTE 默认使用db0的view配方
        );
    }
    selected2Db.value = [];
    myStore.repairButtonFocus(e);
};

// db wise compare
const dbWiseCompList = computed(() =>
    mixedDbList.filter((d) => typeof d.parentId != "string")
);

// const dbWiseCompSnapshotSizesList = computed(() =>
//     dbWiseCompList.value.map((d) =>
//         getImageDimensionsFromBase64Async(
//             myStore.getPrincipalSnapshotOfDashboard(d)
//         )
//     )
// );
const dbWiseCompCellWidth = 60;
const dbWiseCompCellHeight = 60;
const dbWiseCompTitleHeight = 12;
const dbWiseCompMargin = 20;
const dbWiseCompGapH = 20;
const dbWiseCompGapW = 60;
const dbWiseCompColNum = computed(
    () =>
        (props.width - 2 * dbWiseCompMargin + dbWiseCompGapW) /
        (dbWiseCompCellWidth + dbWiseCompGapW)
);
const dbWiseCompRowNum = computed(() =>
    Math.ceil(dbWiseCompList.value.length / dbWiseCompColNum.value)
);
const dbWiseCompAreaHeight = computed(
    () =>
        2 * dbWiseCompMargin +
        dbWiseCompRowNum.value * dbWiseCompCellHeight +
        (dbWiseCompRowNum.value - 1) * dbWiseCompGapH
);

const dbWiseCompLinks = computed(() =>
    dbWiseCompList.value.flatMap((db, i) => {
        const nodes = root.value
            ?.descendants()
            .filter((d) => db.parentId.includes(d.data.id))!;

        const source = {
            x:
                centeredHeight.value +
                x0.value -
                dx.value +
                dbWiseCompMargin +
                dbWiseCompTitleHeight +
                Math.floor(i / dbWiseCompColNum.value) *
                    (dbWiseCompCellHeight + dbWiseCompGapH) +
                dbWiseCompCellHeight / 2,

            y:
                (-dy.value * horizontalPaddingStep) / 2 +
                dbWiseCompMargin +
                (i % dbWiseCompColNum.value) *
                    (dbWiseCompCellWidth + dbWiseCompGapW) +
                dbWiseCompCellWidth / 2,
        };
        const target0 = { x: nodes[0].x, y: nodes[0].y };
        const target1 = { x: nodes[1].x, y: nodes[1].y };
        return [
            { source, target: target0 },
            { source, target: target1 },
        ];
    })
);
const dbWiseCompCurveStrings = computed(() =>
    dbWiseCompLinks.value.map(curveGenerator)
);
</script>

<style scoped></style>
