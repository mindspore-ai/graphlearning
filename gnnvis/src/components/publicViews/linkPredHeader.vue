<template>
    <el-popover
        placement="top-end"
        :trigger="myStore.settingsMenuTriggerMode"
        :width="540"
    >
        <template #reference>
            <setting
                style="
                    height: 1.5em;
                    width: 1.5em;
                    color: var(--el-text-color-regular);
                "
            />
        </template>

        <div style="text-align: start">
            <h3>Link Prediction View Settings</h3>
            <el-divider />

            <span class="setting-item-name"> show neighbors: {{ " " }} </span>
            <el-switch
                v-model="view.isShowNeighbors"
                inactive-text="only selections"
                active-text="selections and its neighbors"
            ></el-switch>
            <br />
            <span class="setting-item-name"> current hops: {{ " " }} </span>
            <el-slider
                v-model="view.currentHops"
                :min="1"
                :max="hops"
                :marks="hopMarks"
            ></el-slider>
            <br />
            <span :style="{ color: 'red', fontWeight: 'bold' }"
                >Warning: modify hops will cause rerender!</span
            >
            <br />
            <br />

            <span class="setting-item-name">
                which types of links are displayed: {{ " " }}
            </span>
            <br />
            <el-button
                color="#555"
                :style="{
                    textDecoration: view.isShowGroundTruth
                        ? 'inherit'
                        : 'line-through',
                }"
                @click="
                    (e:Event) => {
                        if (!view.isShowGroundTruth) {
                            view.isShowTrueAllow = false;
                            view.isShowFalseAllow = false;
                        }
                        view.isShowGroundTruth = !view.isShowGroundTruth;
                        myStore.repairButtonFocus(e);
                    }
                "
                >{{
                    (view.isShowGroundTruth ? "✅️" : "❎") +
                    "Ground Truth (Origin)"
                }}
            </el-button>
            <!-- @change="(v:boolean) => { if (v) { view.isShowTrueAllow = false; view.isShowFalseAllow = false; } }" -->
            <br />
            <el-tooltip
                effect="dark"
                placement="top"
                content="contained in origin graph, and predicted right"
            >
                <el-button
                    @click="
                        (e:Event) => {
                            if (!view.isShowTrueAllow) {
                                view.isShowGroundTruth = false;
                            }
                            view.isShowTrueAllow = !view.isShowTrueAllow;
                            myStore.repairButtonFocus(e);
                        }
                    "
                    color="black"
                    :style="{
                        textDecoration: view.isShowTrueAllow
                            ? 'inherit'
                            : 'line-through',
                    }"
                >
                    {{ (view.isShowTrueAllow ? "✅️" : "❎") + "True Allow" }}
                </el-button>
            </el-tooltip>
            <el-tooltip
                effect="dark"
                placement="top"
                content="contained in origin graph, but predicted wrong"
            >
                <el-button
                    @click="
                        (e:Event) => {
                            if (!view.isShowFalseAllow) {
                                view.isShowGroundTruth = false;
                            }
                            view.isShowFalseAllow = !view.isShowFalseAllow;
                            myStore.repairButtonFocus(e);
                        }
                    "
                    color="red"
                    :style="{
                        textDecoration: view.isShowFalseAllow
                            ? 'inherit'
                            : 'line-through',
                    }"
                >
                    {{ (view.isShowFalseAllow ? "✅️" : "❎") + "False Allow" }}
                </el-button>
            </el-tooltip>
            <br />
            <el-tooltip
                effect="dark"
                placement="top"
                content="not contained in origin graph, recommended and sorted by score"
            >
                <el-button
                    @click="
                        (e:Event) => {
                            view.isShowTrueUnseen = !view.isShowTrueUnseen;
                            myStore.repairButtonFocus(e);
                        }
                    "
                    type="primary"
                    :style="{
                        textDecoration: view.isShowTrueUnseen
                            ? 'inherit'
                            : 'line-through',
                    }"
                >
                    <span>
                        {{
                            (view.isShowTrueUnseen ? "✅️" : "❎") +
                            "True Unseen"
                        }}
                    </span>
                    <span :style="{ fontWeight: 'bold' }">
                        {{ " " }}(Warning: will cause rerender! Only nodes in
                        this dashboard included!)
                    </span>
                </el-button>
                <el-checkbox
                    v-model="view.isShowTrueUnseen"
                    label="True Unseen"
                    border
                    >{{
                        `True Unseen\n(Warning: will cause rerender! 
                       Only nodes in this dashboard included!)`
                    }}
                </el-checkbox>
            </el-tooltip>
            <br />

            <span class="setting-item-name"
                >current true unseen top k(total {{ thisDs.trueUnseenTopK }}):{{
                    " "
                }}
            </span>
            <el-slider
                v-model="view.numTrueUnseen"
                :disabled="!view.isShowTrueUnseen"
                :min="1"
                :step="1"
                show-stops
                :max="thisDs.trueUnseenTopK"
            />
            <br />
            <span class="setting-item-name"> show self loops: {{ " " }} </span>
            <el-switch
                v-model="view.isShowSelfLoop"
                :disabled="view.isShowGroundTruth"
            ></el-switch>
            <br />

            <span class="setting-item-name">node radius {{ " " }} </span>
            <el-slider
                v-model="view.nodeRadius"
                :min="2"
                :step="0.1"
                :max="
                    Math.max(Math.min(view.bodyWidth, view.bodyHeight) / 40, 10)
                "
            />
        </div>
    </el-popover>
</template>

<script setup lang="ts">
import { toRaw, shallowRef, ref, watch, computed, onMounted } from "vue";
import { useMyStore } from "@/stores/store";
import * as d3 from "d3";
import type {
    SingleDashboard,
    NodeView,
    Node,
    NodeCoord,
    LinkPredView,
    CompDashboard,
} from "@/types/types";
import { default as setting } from "@/components/icon/FluentSettings48Regular.vue";
import { ElMessage } from "element-plus";
defineOptions({
    //因为我们没有用一个整体的div包起来，需要禁止透传
    inheritAttrs: false, //NOTE vue 3.3+
});
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
        required: true,
    },
});
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
const view = myStore.getViewByName(db, props.viewName) as NodeView<
    NodeCoord & d3.SimulationNodeDatum
> &
    LinkPredView;
const hops =
    thisDs?.hops ||
    myStore.getDatasetByName((db as CompDashboard).refDatasetsNames[1]).hops ||
    myStore.defaultHops;

view.isShowNeighbors = true;
view.currentHops = hops >= 2 ? 2 : 1;
view.isShowTrueUnseen = false;
view.isShowGroundTruth = false;
view.isShowTrueAllow = true;
view.isShowFalseAllow = true;
view.numTrueUnseen = thisDs.trueUnseenTopK < 2 ? 1 : 2;
view.symbolUnseen = d3.symbolCross;
view.symbolSelection = d3.symbolCircle;
view.nodeRadius = 5;

const hopMarks = computed<Record<number, string>>(() => {
    const ret: Record<number, string> = {};
    for (let i = 1; i <= hops; ++i) {
        ret[i] = i + "";
    }
    return ret;
});
</script>

<style scoped></style>
