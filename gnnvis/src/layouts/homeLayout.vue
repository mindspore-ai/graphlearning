<template>
    <el-affix position="top" :offset="0">
        <div class="home-header"><h1>CorGIE-2 Home</h1></div>
    </el-affix>

    <div class="home-main">
        <!-- <div v-if="isLoading" style="text-align: center">
            Loading Dataset List...
        </div> -->
        <el-table
            v-loading="isLoading"
            :data="data"
            height="90vh"
            style="width: 100%"
            :row-class-name="setTableRowClassName"
            :row-style="() => 'height:80px'"
        >
            <el-table-column prop="date" label="Date" width="180">
            </el-table-column>
            <el-table-column prop="name" label="Name(ID)" width="180">
            </el-table-column>
            <el-table-column prop="task" label="Task Type"> </el-table-column>
            <el-table-column prop="graph" label="Graph"> </el-table-column>
            <el-table-column prop="model" label="Training Model(Network)">
            </el-table-column>
            <el-table-column label="Operation">
                <template #default="scope">
                    <template v-if="!selectedName">
                        <RouterLink
                            :to="`/single/${scope.row.name}`"
                            :replace="true"
                        >
                            <el-button size="small">Analyze</el-button>
                        </RouterLink>
                        <template
                            v-if="
                                data.filter(
                                    (d) =>
                                        d.graph === scope.row.graph &&
                                        d.task === scope.row.task
                                ).length > 1
                            "
                        >
                            <!-- && d.model === scope.row.model -->
                            <el-divider direction="vertical"></el-divider>
                            <el-button
                                size="small"
                                @click="() => setSelectedName(scope.row.name)"
                                >Compare</el-button
                            ></template
                        >
                    </template>
                    <template v-else>
                        <el-button
                            v-if="scope.row.name === selectedName"
                            size="small"
                            type="danger"
                            @click="() => setSelectedName('')"
                            >Cancel</el-button
                        >
                        <RouterLink
                            v-else
                            :to="`/compare/${selectedName}/${scope.row.name}`"
                            :replace="true"
                            ><el-button size="small">+</el-button></RouterLink
                        ></template
                    >
                </template>
            </el-table-column>
        </el-table>

        <p v-if="error">{{ error }}</p>
    </div>
    <!-- <el-affix position="bottom" :offset="0"> -->
    <!-- <div class="home-footer">footer</div> -->
    <!-- </el-affix> -->
</template>

<script setup lang="ts">
import { useMyStore } from "@/stores/store";
import { ref, computed } from "vue";
// import { useFetch } from "@vueuse/core";
import { baseUrl } from "../api/api";
import type { Type_TaskTypes, UrlData } from "@/types/types";
const myStore = useMyStore();
const isLoading = ref(false);
const error = ref<Error>();
const myFetch = async () => {
    try {
        isLoading.value = true;
        error.value = undefined;
        const res = await fetch(baseUrl + "list.json");
        // console.log("in homeLayout, res", res);
        if (res.ok) {
            const dataJson = await res.json();
            // datasetList.value = dataJson.list;

            myStore.urlList = dataJson.list.map(
                (d: {
                    name: string;
                    task: Type_TaskTypes;
                    date?: Date;
                    graph: string;
                    model: string;
                }) => {
                    if (d.date) return d;
                    else {
                        const date = new Date();
                        const year = date
                            .getFullYear()
                            .toString()
                            .padStart(4, "0");
                        const month = (date.getMonth() + 1)
                            .toString()
                            .padStart(2, "0");
                        const day = date.getDate().toString().padStart(2, "0");
                        const hour = date
                            .getHours()
                            .toString()
                            .padStart(2, "0");
                        const minute = date
                            .getMinutes()
                            .toString()
                            .padStart(2, "0");
                        const second = date
                            .getSeconds()
                            .toString()
                            .padStart(2, "0");
                        const formattedDate = `${year}-${month}-${day}-${hour}-${minute}-${second}`;
                        return { ...d, date: formattedDate };
                    }
                }
            );
        } else throw new Error("失败了！");
    } catch (e) {
        console.warn(e);
        error.value = e as Error;
    } finally {
        isLoading.value = false;
    }
};
myFetch();
const data = computed(() => {
    if (selectedName.value) {
        const { graph, task } = myStore.urlList.find(
            (d) => d.name === selectedName.value
        ) || { graph: "", model: "", task: "" };
        return myStore.urlList.filter(
            (d: UrlData) => d.graph === graph && d.task === task
        );
    } else {
        return myStore.urlList;
    }
});

const selectedName = ref("");
const setSelectedName = (name: string) => {
    selectedName.value = name;
};
const setTableRowClassName = (obj: { row: UrlData; rowIndex: number }) => {
    const { row, rowIndex } = obj;
    if (row.name === selectedName.value) return "selected-row";
    else {
        if (rowIndex & 1) return "";
        else return "striped-row";
    }
};
</script>

<style>
/* 不能设为scoped， 否则row不生效*/
/* .el-header {
    flex: 0 0 auto;
}
.el-main {
    flex: 1 1 auto;
}
.el-footer {
    flex: 0 0 auto;
} */
.el-table .striped-row {
    background-color: #fafafa;
}
.el-table .selected-row {
    background-color: var(--el-color-success-light-5);
}
.home-header,
.home-footer {
    width: 100%;
    --header-footer-height: 60px;
    height: var(--header-footer-height);
    z-index: 8888;

    box-sizing: border-box; /*fix 20*/
    padding: 0 20px;

    background-color: rgb(102, 177, 255);
    color: var(--el-text-color-primary);

    display: flex;
    justify-content: center;
    align-items: center;
}
/* .header {
    position: fixed;
    top: 0;
    left: 0;
} */
.home-footer {
    position: fixed;
    top: calc(100% - var(--header-footer-height));
}
</style>
