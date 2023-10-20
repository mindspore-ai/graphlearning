<script setup lang="ts">
import { RouterView } from "vue-router";
import { watch, onUnmounted } from "vue";
import { useMyStore } from "./stores/store";

const myStore = useMyStore();
watch(
    () => myStore.globalClearSelMode,
    (newV) => {
        console.log("myStore.globalClearSelMode changed to", newV);
        myStore.singleDashboardList.forEach(
            (d, i, arr) => (arr[i].clearSelMode = newV)
        );
        myStore.compDashboardList.forEach(
            (d, i, arr) => (arr[i].clearSelMode = newV)
        );
    },
    { immediate: true }
);
watch(
    () => myStore.globalWhenToResizeMode,
    (newV) => {
        console.log("myStore.globalWhenToResizeMode changed to", newV);
        myStore.singleDashboardList.forEach(
            (d, i, arr) => (arr[i].whenToRescaleMode = newV)
        );
        myStore.compDashboardList.forEach(
            (d, i, arr) => (arr[i].whenToRescaleMode = newV)
        );
    },
    { immediate: true }
);
watch(
    () => myStore.globalLabelType,
    (newV) => {
        console.log("myStore.globalWhenToResizeMode changed to", newV);
        myStore.singleDashboardList.forEach(
            (d, i, arr) => (arr[i].labelType = newV)
        );
        myStore.compDashboardList.forEach(
            (d, i, arr) => (arr[i].labelType = newV)
        );
    },
    { immediate: true }
);
onUnmounted(() => {
    console.warn("App unmounted!");
    myStore.testWatcher();
});
</script>

<template>
    <RouterView />
</template>

<style scoped></style>
