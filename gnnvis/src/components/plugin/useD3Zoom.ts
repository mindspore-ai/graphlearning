import type { ReadonlyRefOrGetter } from "@vueuse/core";
import * as d3 from "d3";
import {
    ref,
    unref,
    watch,
    onMounted,
    onUnmounted,
    type Ref,
    type MaybeRefOrGetter,
} from "vue";

export const useD3Zoom = (
    domRef: Ref<SVGElement | null>,
    zoomAreaQuery: string | undefined, //usually svg itself
    width: ReadonlyRefOrGetter<number>,
    height: ReadonlyRefOrGetter<number>,
    calcApplyExtentByWH = (
        w: number,
        h: number
    ): [[number, number], [number, number]] => [
        [0, 0],
        [w, h],
    ], //function
    scaleExtent: MaybeRefOrGetter<[number, number]> = [1 / 2, 16],
    calcTransExtentByWH = calcApplyExtentByWH,
    isPanEnabled: MaybeRefOrGetter<boolean> = false, //usually the invert of isBrushEnabled
    debugViewName = "",
    isDisableWheel = false
) => {
    //NOTE resize 改变w,h时，会自动反映到props.width, props.height，因此，用函数式编程以动态相应w、h即可。从而和resize逻辑解耦
    const transformRef = ref(d3.zoomIdentity);
    const enablePanFuncRef = ref(() => {});
    const disablePanFuncRef = ref(() => {});
    const resetZoomFuncRef = ref(() => {});

    const zoom = d3
        .zoom<
            SVGElement, //Element type
            unknown //Datum type
        >()
        .extent(
            calcApplyExtentByWH(
                typeof width === "function" ? width() : unref(width),
                typeof height === "function" ? height() : unref(height)
            )
        )
        .scaleExtent(
            typeof scaleExtent === "function"
                ? scaleExtent()
                : unref(scaleExtent)
        )
        .translateExtent(
            calcTransExtentByWH(
                typeof width === "function" ? width() : unref(width),
                typeof height === "function" ? height() : unref(height)
            )
        )
        .on("zoom", (event: d3.D3ZoomEvent<SVGElement, null>) => {
            transformRef.value = event.transform;
        });

    watch([width, height], ([newWidth, newHeight]) => {
        zoom.extent(calcApplyExtentByWH(newWidth, newHeight)).translateExtent(
            calcTransExtentByWH(newWidth, newHeight)
        );
    });
    onMounted(() => {
        console.log(`in ${debugViewName},in useD3Zoom, Mounted!`);
        const sel = zoomAreaQuery
            ? d3.select(domRef.value).select(zoomAreaQuery)
            : d3.select(domRef.value);

        if (sel.node()) {
            // first time
            if (unref(isPanEnabled)) {
                zoom.filter(
                    (event) =>
                        (!event.ctrlKey ||
                            (isDisableWheel
                                ? false
                                : event.type === "wheel")) &&
                        !event.button //default
                );
            } else {
                zoom.filter((event) =>
                    isDisableWheel ? false : event.type === "wheel"
                ); //no pan
            }

            // bind
            (sel as d3.Selection<SVGElement, unknown, null, undefined>).call(
                zoom,
                d3.zoomIdentity
            ); //always

            // expose `ref`s
            disablePanFuncRef.value = () => {
                zoom.filter((event) =>
                    isDisableWheel ? false : event.type === "wheel"
                ); //no pan
            };
            enablePanFuncRef.value = () => {
                zoom.filter(
                    (event) =>
                        (!event.ctrlKey ||
                            (isDisableWheel
                                ? false
                                : event.type === "wheel")) &&
                        !event.button //default
                );
            };
            resetZoomFuncRef.value = () => {
                const tempExtent = calcApplyExtentByWH(
                    typeof width === "function" ? width() : unref(width),
                    typeof height === "function" ? height() : unref(height)
                );
                (
                    sel.transition().duration(750) as d3.Transition<
                        SVGElement,
                        unknown, //Datum
                        null, //Parent element
                        undefined //Parent datum
                    >
                ).call(
                    zoom.transform,
                    d3.zoomIdentity,
                    d3
                        .zoomTransform(sel.node() as SVGElement)
                        .invert([
                            (tempExtent[0][0] + tempExtent[1][0]) / 2,
                            (tempExtent[0][1] + tempExtent[1][1]) / 2,
                        ])
                );
            };
        }
    });
    onUnmounted(() => {
        console.log(`in ${debugViewName},in useD3Zoom, Unmounted!`);
        disablePanFuncRef.value = () => {};
        enablePanFuncRef.value = () => {};
        zoom.on("zoom", null);
    });
    return {
        transformRef,
        enablePanFuncRef,
        disablePanFuncRef,
        resetZoomFuncRef,
    };
};
