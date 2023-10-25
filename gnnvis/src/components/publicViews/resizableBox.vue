<template>
    <div ref="el" :class="[db.isRepresented ? 'non-resize' : 'resize', 'box']">
        <div class="resizable-box-head" id="resizableBoxHeader">
            <slot name="header" ref="headChild1"> </slot>
        </div>
        <div
            :class="[
                'resizable-box-body',
                `resizable-box-bodies-${props.dbId}`,
            ]"
            ref="body"
        >
            <slot ref="child"></slot>
        </div>
    </div>
</template>

<script setup lang="ts">
import { useElementBounding } from "@vueuse/core";
import {
    ref,
    computed,
    watch,
    onMounted,
    onUnmounted,
    onBeforeUnmount,
    nextTick,
} from "vue";
import { useMyStore } from "@/stores/store";

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
});
const myStore = useMyStore();
const viewDef = myStore.getViewByName(props.dbId, props.viewName)!;
const db = myStore.getTypeReducedDashboardById(props.dbId)!;
////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION initial size
// const computedWidthStr = computed(() => viewDef.initialWidth + "px");
// const computedHeightStr = computed(() => viewDef.initialHeight + "px");

const widthStr = ref(viewDef.initialWidth + "px");
const heightStr = ref(viewDef.initialHeight + "px");
watch(
    [() => viewDef.initialWidth, () => viewDef.initialHeight],
    ([newW, newH]) => {
        widthStr.value = newW + "px";
        heightStr.value = newH + "px";
    },
    { immediate: true }
);
const borderWidth = 1;
const borderWidthStr = computed(() => borderWidth + "px");
// console.log(
//     "in resizableBox,",
//     props.viewName,
//     "onCreated, computedWidthStr & computedHeightStr:",
//     computedWidthStr.value,
//     computedHeightStr.value
// );
////////////////// !SECTION initial size
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION resize size calc
const el = ref<HTMLDivElement | null>(null);
const { x, y, top, right, bottom, left, width, height } =
    useElementBounding(el);
//NOTE 和box-sizing无关。这个height一直将border算在内

// const bodyHeight = ref(0); //NOTE 我们将body的size提取到store，就不用slot绑定属性了！·
// const bodyWidth = ref(0);
defineExpose({
    boundingWidth: width,
    boundingHeight: height,
    widthStr,
    heightStr,
});

const mountedFlag = ref(false);
onMounted(async () => {
    mountedFlag.value = true;
    // console.log(
    //     "in resizableBox,",
    //     props.viewName,
    //     "onMounted\nuseBounding width & height:",
    //     width.value,
    //     height.value
    // );
});
onUnmounted(() => {
    mountedFlag.value = false;
});
watch([width, height], ([newW, newH]) => {
    if (mountedFlag.value) {
        const headerElement = el.value?.querySelector("#resizableBoxHeader");
        const headerStyle = window.getComputedStyle(headerElement!);

        console.log(
            "in resizableBox viewName:",
            props.viewName,
            "watch width & height, got new",
            newW,
            newH,
            "\nheaderElement.clientHeight",
            headerElement?.clientHeight, //NOTE 此方案永远不包括border,当border-box时，算的是总尺寸-border，当content-box，算的是content。但是包括padding
            "\nheaderStyle.height",
            headerStyle.height //NOTE  获取的是css，恒为32,且不包括padding
        );

        // if (db.isRepresented) {
        //     widthStr.value = newW + "px";
        //     heightStr.value = newH + "px";
        // }

        // bodyHeight.value = newH - parseFloat(headerStyle.height) - 3 * borderWidth;
        // bodyHeight.value =
        //     newH - (headerElement?.clientHeight || 32) - 3 * borderWidth;

        // bodyWidth.value = newW - 2 * borderWidth;

        // viewDef.bodyHeight = newH - parseFloat(headerStyle.height) - 3 * borderWidth;
        viewDef.bodyHeight = Math.max(
            newH - (headerElement?.clientHeight || 40) - 3 * borderWidth,
            0
        );
        viewDef.bodyWidth = Math.max(newW - 2 * borderWidth, 0);
    }
});
////////////////// !SECTION resize size calc
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION resize end
const resizeTimer = ref();
// const resizeEndSignal = ref(false);//已经提取到store了
watch([width, height], () => {
    clearTimeout(resizeTimer.value);
    resizeTimer.value = setTimeout(() => {
        // resizeEndSignal.value = !resizeEndSignal.value;
        viewDef.resizeEndSignal = !viewDef.resizeEndSignal;
    }, 400); // adjust the timeout as needed
});
////////////////// !SECTION resize end
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////// SECTION restore size
watch(
    // () => props.restoreViewsSizesSignal,
    () => db.restoreViewsSizesSignal,
    async () => {
        console.log("in resizable, props.restoreViewsSizesSignal change!");
        el.value!.style.width = viewDef.initialWidth + "px";
        el.value!.style.height = viewDef.initialWidth + "px";
    },
    { immediate: false }
);
////////////////// !SECTION restore size
////////////////////////////////////////////////////////////////////////////////

onBeforeUnmount(() => {
    console.warn(
        "in Resizable Box of view:",
        props.viewName,
        ", onBeforeUnmount!"
    );
    myStore
        .getViewByName(props.dbId, props.viewName)
        ?.onBeforeUnmountCallbacks?.forEach((d) => d());
});

//NOTE 在slot中不能用child的ref, 写了也是白写

// 以下方案有难度。暂时使用document.querySelectorAll
// const body = ref<HTMLDivElement | null>(null);
// watch(
//     body,
//     (newV) => {
//         console.log(
//             "in",
//             props.viewName,
//             "in resizableBox, body ref changed",
//             newV
//         );
//     },
//     { deep: true, immediate: true }
// );
// defineExpose({ body });
</script>

<style scoped>
.resize {
    resize: both;
}
.non-resize {
    resize: none;
    /* width: 100%;
    height: 100%; */
}
.box {
    width: v-bind(widthStr);
    height: v-bind(heightStr);
    overflow: hidden;
    /* width: v-bind("viewDef.initialWidth + 'px'"); */

    /* box-sizing: border-box; */
    box-sizing: content-box;

    border-radius: 4px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.12), 0 0 6px rgba(0, 0, 0, 0.04);
    border: solid rgba(0, 0, 0, 0.12);
    border-width: v-bind(borderWidthStr);
}
.resizable-box-head {
    /* overflow: visible; */
    min-height: 2em;
    padding: 0.25em 1em;
    /* width: calc(100% - 2em); */
    /* box-sizing: border-box; */
    box-sizing: content-box;
    border-bottom: solid rgba(0, 0, 0, 0.12);
    border-bottom-width: v-bind(borderWidthStr);
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.resizable-box-body {
    border: none;
    padding: 0;
    margin: 0;
}
</style>
