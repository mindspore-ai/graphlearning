import { ref, watch, computed, unref } from "vue";
import type { Ref } from "vue";

export const useResize = (
    resizeEndSignal: Ref<boolean> | (() => boolean),
    isRescaleOnResizeEnd: Ref<boolean> | (() => boolean),
    // points: Ref<Array<any>>, //ref
    width: Ref<number> | (() => number),
    height: Ref<number> | (() => number),
    rescaleCoordsCaller = (): any => {}, // (coordsRef)=>specificRescaleFunc(coordsRef.value)
    debugViewName = ""
) => {
    //NOTE 其实 resize的本质就是scale,水平方向放大newWidth/oldWidth倍,竖直方向放大newHeight/oldHeight倍

    const widthScaleRatio = ref(1);
    const heightScaleRatio = ref(1);

    const extractedWidth = computed(() =>
        typeof width === "function" ? width() : unref(width)
    );
    const extractedHeight = computed(() =>
        typeof height === "function" ? height() : unref(height)
    );

    const lastOldWidth = ref(extractedWidth.value); //脱钩
    const lastOldHeight = ref(extractedHeight.value);
    watch(
        resizeEndSignal, //在resizeEnd时更新
        () => {
            if (
                typeof isRescaleOnResizeEnd === "function"
                    ? isRescaleOnResizeEnd()
                    : unref(isRescaleOnResizeEnd)
            ) {
                widthScaleRatio.value =
                    extractedWidth.value / lastOldWidth.value;
                heightScaleRatio.value =
                    extractedHeight.value / lastOldHeight.value;
                rescaleCoordsCaller();
            }

            lastOldWidth.value = extractedWidth.value; //相当于延迟记录上一次的
            lastOldHeight.value = extractedHeight.value; //相当于延迟记录上一次的
        },
        { flush: "post" }
    );
    watch(
        [width, height],
        ([newWidth, newHeight], [oldWidth, oldHeight]) => {
            console.log(
                `in ${debugViewName} useResize, watch size, \nnew: ${newWidth}, ${newHeight}, \nold:${oldWidth}, ${oldHeight}`,
                "newRatio will be:",
                oldWidth > 0 && newWidth > 0 ? newWidth / oldWidth : 1,
                oldHeight > 0 && newHeight > 0 ? newHeight / oldHeight : 1
            );

            //即时更新 //虽然computed也可以，但是这里必须用watch，因为isRescaleOnResizeEnd和width/height两种状态是耦合的

            if (
                typeof isRescaleOnResizeEnd === "function"
                    ? !isRescaleOnResizeEnd()
                    : !unref(isRescaleOnResizeEnd)
            ) {
                widthScaleRatio.value =
                    oldWidth > 0 && newWidth > 0 ? newWidth / oldWidth : 1; //REVIEW -  意外情况使用0或1？
                heightScaleRatio.value =
                    oldHeight > 0 && newHeight > 0 ? newHeight / oldHeight : 1;
                rescaleCoordsCaller();
            }
        },
        { flush: "post" }
    );
    return { widthScaleRatio, heightScaleRatio };
};
