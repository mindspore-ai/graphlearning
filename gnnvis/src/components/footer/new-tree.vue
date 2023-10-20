<template>
    <div>
        <LoadingComp v-if="isLoading" text="calculating layout" />
        <template v-else>
            <span
                >select one db to go to, select two dbs to compare
                {{ " " }}
            </span>
            <el-button
                @click="(e:Event) => handleGotoClick(e, selected2Db[0])"
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
            <svg :width="props.width" :height="props.height">
                <g class="group">
                    <rect
                        v-for="(d, i) in groups"
                        :key="i"
                        :x="d.bounds.x"
                        :y="d.bounds.y"
                        :width="d.bounds.width()"
                        :height="d.bounds.height()"
                        stroke="gold"
                        stroke-width="2"
                        fill="red"
                        fill-opacity="0.3"
                    ></rect>
                </g>
                <g class="links">
                    <line
                        v-for="d in links"
                        :key="d.eid"
                        :x1="d.source.x"
                        :y1="d.source.y"
                        :x2="d.target.x"
                        :y2="d.target.y"
                        :stroke="d.isDbWiseComp ? 'green' : 'black'"
                        stroke-width="2"
                    ></line>
                </g>
                <g class="nodes">
                    <circle
                        @click="() => handleDbClick(d.id)"
                        v-for="d in nodes"
                        :key="d.id"
                        :cx="d.x"
                        :cy="d.y"
                        r="5"
                        :fill="d.isDbWiseComp ? 'green' : 'black'"
                    ></circle>
                </g>
            </svg>
        </template>
    </div>
</template>

<script setup lang="ts">
import { ref, watch, computed } from "vue";
import { useMyStore } from "../../stores/store";

import type {
    CompDashboard,
    Dashboard,
    RecentDb,
    SingleDashboard,
    Type_GraphId,
    Type_LinkId,
    Type_NodeId,
    View,
    DbWiseComparativeDb,
} from "@/types/types";
import { isDbWiseComparativeDb } from "@/types/typeFuncs";

import LoadingComp from "../state/Loading.vue";

import { getImageDimensionsFromBase64Async } from "@/utils/otherUtils";

import { nanoid } from "nanoid";
import * as cola from "webcola";
import * as d3 from "d3";

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
const myStore = useMyStore();

const isLoading = ref(true);
declare interface NonDbWiseCompDb extends Dashboard {
    parentId: string;
}
const mixedDbList = props.isSingle
    ? myStore.singleDashboardList
    : myStore.compDashboardList;
const dbList = computed<Dashboard[]>(() =>
    props.isSingle
        ? myStore.singleDashboardList.filter((db) => !isDbWiseComparativeDb(db))
        : myStore.compDashboardList.filter((db) => !isDbWiseComparativeDb(db))
);
const dbWiseCompList = computed(() =>
    props.isSingle
        ? myStore.singleDashboardList.filter((db) => isDbWiseComparativeDb(db))
        : myStore.compDashboardList.filter((db) => isDbWiseComparativeDb(db))
);
const nodes = computed(() =>
    mixedDbList.map((db) => ({
        id: db.id,
        name: db.name,
        isDbWiseComp: isDbWiseComparativeDb(db),
    }))
);
const nodeIdMapIndex = computed<Record<Dashboard["id"], number>>(() =>
    nodes.value.reduce(
        (acc, cur, curI) => ({
            ...acc,
            [cur.id]: curI,
        }),
        {}
    )
);
const links = computed(() => {
    const ret: {
        source: number; //index
        target: number; //index
        isDbWiseComp: boolean;
        eid: Type_LinkId;
    }[] = [];
    let eid = 0;
    for (const db of mixedDbList) {
        if (db.parentId) {
            if (!isDbWiseComparativeDb(db)) {
                const srcIndex = nodeIdMapIndex.value[db.parentId as string];
                const tgtIndex = nodeIdMapIndex.value[db.id];
                ret.push({
                    source: srcIndex,
                    target: tgtIndex,
                    isDbWiseComp: false,
                    eid: String(eid++),
                });
            } else {
                const srcIndex0 = nodeIdMapIndex.value[db.parentId[0]];
                const srcIndex1 = nodeIdMapIndex.value[db.parentId[1]];
                const tgtIndex = nodeIdMapIndex.value[db.id];
                ret.push({
                    source: srcIndex0,
                    target: tgtIndex,
                    isDbWiseComp: true,
                    eid: String(eid++),
                });
                ret.push({
                    source: srcIndex1,
                    target: tgtIndex,
                    isDbWiseComp: true,
                    eid: String(eid++),
                });
            }
        }
    }
    return ret;
});
const treeFn = d3
    .stratify<{
        id: Dashboard["id"];
        parentId: Exclude<Dashboard["parentId"], string[]>;
    }>()
    .id((d) => d.id)
    .parentId((d) => d.parentId);

const horizontalPaddingStep = 1;
const dx = computed(
    () =>
        props.width /
        (treeFn(dbList.value as NonDbWiseCompDb[]).height +
            2 * horizontalPaddingStep)
);
const constraints = computed(
    () => {
        const ret = [];
        for (const link of links.value) {
            ret.push({
                axis: "x",
                left: link.source,
                right: link.target,
                gap: link.isDbWiseComp ? dx.value * 0 : dx.value * 0.7,
            });
            // if (link.isDbWiseComp) {
            //     ret.push({
            //         axis: "y",
            //         left: link.source,
            //         right: link.target,
            //         gap: 0,
            //     });
            // }
        }
        return ret;
    }
    // links.value.map((link) => ({
    //     axis: "x",
    //     left: link.source,
    //     right: link.target,
    //     gap: link.isDbWiseComp ? 0 : dx.value * 0.7,
    // }))
);

const groups = computed(() =>
    dbWiseCompList.value.map((db) => ({
        leaves: [
            nodeIdMapIndex.value[db.id],
            nodeIdMapIndex.value[db.parentId[0]],
            nodeIdMapIndex.value[db.parentId[1]],
        ],
        padding: 1,
    }))
);
const layout = cola
    .d3adaptor(d3)
    .avoidOverlaps(true)
    .size([props.width, props.height])
    .symmetricDiffLinkLengths(20)
    .stop();
const tickCount = ref(0);
watch(
    [nodes, links, constraints, groups],
    ([newNodes, newLinks, newConstraints, newGroups]) => {
        const p = new Promise<void>((resolve, reject) => {
            isLoading.value = true;
            tickCount.value = 0;
            setTimeout(() => {
                layout
                    .nodes(newNodes as cola.InputNode[])
                    .links(newLinks)
                    .flowLayout("x", (d: (typeof newNodes)[number]) =>
                        d.isDbWiseComp ? dx.value * 0 : dx.value * 0.8
                    )
                    .constraints(constraints.value)
                    .groups(newGroups) //NOTE 是cola类型写得不对
                    .on("tick", () => {
                        console.log("in history tree layout calc, ticking");
                        if (++tickCount.value > 1000) {
                            // resolve();
                        }
                    })
                    .on("end", () => {
                        console.log("in history tree layout calc, finished");
                        resolve();
                    })
                    .start(20, 20, 0);
            }, 1500);
        });
        p.then(() => {
            isLoading.value = false;
        });
    },
    { immediate: true, deep: true }
);

const currentDashboardId = computed(() =>
    props.isSingle
        ? myStore.recentSingleDashboardList.at(-1)?.id || ""
        : myStore.recentCompDashboardList.at(-1)?.id || ""
);
const selected2Db = ref<string[]>([]);
const handleDbClick = (id: string) => {
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
</script>

<style scoped></style>
