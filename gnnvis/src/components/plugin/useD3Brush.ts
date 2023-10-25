import type { NodeCoord } from "@/types/types";
import * as d3 from "d3";
import {
    type Ref,
    ref,
    unref,
    watch,
    computed,
    onMounted,
    onUnmounted,
    onBeforeUnmount,
    isRef,
} from "vue";
/**
 * Bind d3.brush() to an svg <g> element, and this will create a rect mask,\
 * the area, extent, points can be customized,\
 * the selected dict will be modified through passed params: Ref or func\
 * return control funcs as ref, such as enable and disable et al.\
 * NOTE we can separate the logics of brush and zoom, namely, we have 2 funcs to enable pan or brush in origin component but now we can use 4 funcs\
 * NOTE the side effect of resize can be extracted and then passed as params: widthScaleRatio, heightScaleRatio,\
 * \
 * how do we provide brush functionality?\
 * we can envisage:\
 *  register(in this use impl): enableRegister.value = enableFunc  // the param should be a ref or reactive.key,\
 *  directly assign so we can call : enableRegister() or enableRegister.value()\
 * or:\
 *  register(in this use impl): enableRegister(enableFunc);\
 *  actually impl: (enableFunc)=> something= enableFunc; and call: something()\
 * instead:\
 *  we return ref(func), and let the caller decide how to use them (inject, props, emits, pinia, click, et al)\
 */
export const useD3Brush = <T extends NodeCoord>(
    brushRef: Ref<SVGGElement | null>,
    applyExtent: () => [[number, number], [number, number]], //function
    innerExtent: () => [[number, number], [number, number]], //function
    points: Ref<Array<T>> | (() => Array<T>),
    clearDictFunc = () => {},
    selDictFuncOrRef: ((id: string) => void) | Ref<Record<string, boolean>> = (
        id: string
    ) => {
        console.log(id, " brushed!"); //do something using id
    },
    isBrushable:
        | ((x: number, y: number) => boolean)
        | Ref<(x: number, y: number) => boolean> = () => true,
    isManualClear: boolean | Ref<boolean> | (() => boolean) = false,
    isBrushEnabled: boolean | Ref<boolean> | (() => boolean) = true,
    getX = (d: T) => d.x,
    getY = (d: T) => d.y,
    widthScaleRatio: Ref<number>,
    heightScaleRatio: Ref<number>,
    debugViewName = ""
) => {
    const enableBrushFuncRef = ref(() => {});
    const disableBrushFuncRef = ref(() => {});
    const hideBrushRectFuncRef = ref(() => {}); //when user clears selection in brush mode, then the rect should be hidden.

    const localSearchFlag = ref(true); //sometimes we have to forbid search to improve UI performance

    const brushablePoints = computed(() => {
        const p = isRef(points)
            ? unref(points)
            : (points as () => Array<any>)();
        return p.filter((d) => unref(isBrushable)(d.x, d.y));
    });

    // console.log( `in ${debugViewName}, in useD3brush, isRef(points)${isRef(points)}`); //true
    // console.log(`in ${debugViewName}, in useD3brush, points[0]${points[0]}`); //undefined
    // console.log( `in ${debugViewName}, in useD3brush, unref(points)[0]${ unref(points)[0] }`); //[object Object]
    // console.log( `in ${debugViewName}, in useD3brush, brushablePoints.value[0]${brushablePoints.value[0]}`);
    // watch(
    //     points,
    //     (newV) => {
    //         console.warn(
    //             "in useD3Brush, in",
    //             debugViewName,
    //             "points changed!",
    //             newV
    //         );
    //     },
    //     { deep: true }
    // );

    const quadtree = computed(() =>
        d3
            .quadtree<T>()
            .extent(innerExtent())
            .x((d) => getX(d))
            .y((d) => getY(d)) // NOTE xy first and then add points, otherwise error
            .addAll(brushablePoints.value || [])
    );

    const brushEnd: (e: d3.D3BrushEvent<T>) => void = (e) => {
        //NOTE the param is actually only one, which is different from d3 doc on their official website
        // @types/d3 reads: (this: SVGGElement, event: any, d: T) => void
        //LINK https://github.com/d3/d3-brush/blob/v3.0.0/README.md#brush_on

        // console.log("in d3 brush end, params", e);
        // console.log("in d3 brush end, this is ", this); //undefined
        const ext = e.selection;
        if (ext) {
            if (localSearchFlag.value)
                search(
                    quadtree.value,
                    ext as [[number, number], [number, number]]
                );
        }
        localSearchFlag.value = true; //make search available next time
    };
    const brushStart = () => {
        if (
            typeof isManualClear === "function"
                ? !isManualClear()
                : !unref(isManualClear)
        ) {
            clearDictFunc();
        }
    };

    const brush = d3
        .brush()
        .on("start", brushStart)
        .on("end", brushEnd)
        .extent(applyExtent());
    //NOTE we can change the extent of quadtree manually or use `computed`
    //      but when we change brush.extent, we should also `call` to bind it to dom element
    //NOTE unlike quadtree which only has data features,
    //      brush has both data and UI logics, thus, we change the UI rect of brush extent reactively
    watch(applyExtent, (newV) => {
        console.warn(
            "in",
            debugViewName,
            "in useD3Brush, in watch applyExtent, we will apply brush.extent to",
            newV
        );

        brush.extent(newV); //change the extent reactively (for quadtree, we've already used `computed`)

        const gBrush = d3.select(brushRef.value);
        if (gBrush.node()) {
            (
                gBrush as d3.Selection<SVGGElement, unknown, null, undefined>
            ).call(brush);
        }
    });

    const search = (
        quadtree: d3.Quadtree<T>,
        [[x0, y0], [x3, y3]]: [[number, number], [number, number]]
    ) => {
        console.log(
            "in useD3Brush, in",
            debugViewName,
            "called search!",
            x0,
            y0,
            x3,
            y3
        );
        quadtree.visit((node, x1, y1, x2, y2) => {
            //node: not-leaf [1,2,3,4] or leaf {data: , next:}
            // console.log("in visit, node:", node);
            if (!node.length) {
                // leaf nodes
                do {
                    const {
                        data: d, //{id,x,y}
                        data: { x, y }, //双解构赋值
                    } = node as d3.QuadtreeLeaf<T>;
                    if (x >= x0 && x < x3 && y >= y0 && y < y3) {
                        if (typeof selDictFuncOrRef === "function") {
                            selDictFuncOrRef(d.id);
                        } else {
                            selDictFuncOrRef.value[d.id] = true;
                            //NOTE passed params: points and selDictFuncOrRef(how to modify selection) are different conceptions,
                            // check if they both have reactive features.
                        }
                    }
                } while (node.next && (node = node.next)); //不为undefined继续
            }
            return x1 >= x3 || y1 >= y3 || x2 < x0 || y2 < y0;
        });
        if (typeof selDictFuncOrRef !== "function") {
            console.log(
                `in ${debugViewName}, in useD3brush, search ret`,
                selDictFuncOrRef.value
            );
        }
    };

    // this watcher is used to make the rect change size corresponding to the svg size
    watch([widthScaleRatio, heightScaleRatio], ([newWR, newHR]) => {
        const gBrush = d3.select(brushRef.value);

        if (gBrush.node()) {
            const rect = d3.brushSelection(gBrush.node() as SVGGElement);
            console.log(
                "in",
                debugViewName,
                "in useD3Brush,in watch Ratios, ",
                "\ngot newWR, newHR",
                newWR,
                newHR,
                "\ngot rect",
                rect,
                "if the rect is not null, we will change rect"
            );
            if (rect) {
                const [[x0, y0], [x1, y1]] = rect as [
                    [number, number],
                    [number, number]
                ];

                brush.on("end", null).on("start", null); //forbid `start` & `end` to improve performance, since the source code of `move` invokes `start` and `end`

                (
                    gBrush as d3.Selection<
                        SVGGElement,
                        unknown,
                        null,
                        undefined
                    >
                ).call(brush.move, (datum, i, node) => {
                    if (newWR != Infinity && newHR != Infinity) {
                        return [
                            [x0 * newWR, y0 * newHR],
                            [x1 * newWR, y1 * newHR],
                        ];
                    } else {
                        return [
                            //if Infinity, then 0 * Inf = NaN, then there'll be bugs, esp. when switch dashboards
                            [0, 0],
                            [0, 0],
                        ];
                    }
                });

                brush.on("end", brushEnd).on("start", brushStart); // rebind
            }
        }
    });

    onMounted(() => {
        console.log(`in ${debugViewName},in useD3Brush, Mounted!`);

        enableBrushFuncRef.value = () => {
            const gBrush = d3.select(brushRef.value);
            if (gBrush.node()) {
                gBrush.style("display", "inherit");
                gBrush.attr("pointer-events", "all");

                (
                    gBrush as d3.Selection<
                        SVGGElement,
                        unknown,
                        null,
                        undefined
                    >
                ).call(brush); // rebind
            }
        };

        disableBrushFuncRef.value = () => {
            const gBrush = d3.select(brushRef.value);
            if (gBrush.node()) {
                gBrush.attr("pointer-events", "none");
                gBrush.style("display", "none");
                //NOTE we implement "disable" by hiding rather than removing
                // which facilitates users by letting them see the old rect when they enable brush again
            }
        };

        hideBrushRectFuncRef.value = () => {
            const gBrush = d3.select(brushRef.value);
            if (gBrush.node()) {
                if (d3.brushSelection(gBrush.node() as SVGGElement)) {
                    // NOTE it's not enough to only check whether it's in brush mode
                    //we should also check whether the rect exists
                    (
                        gBrush as d3.Selection<
                            SVGGElement,
                            unknown,
                            null,
                            undefined
                        >
                    ).call(brush.clear);
                }
            }
        };

        if (typeof isBrushEnabled === "function") {
            if (isBrushEnabled()) enableBrushFuncRef.value();
        } else {
            if (unref(isBrushEnabled)) {
                enableBrushFuncRef.value();
                // first time, is brush mode or zoom mode
                // if (zoom) zoom.filter((event) => event.type === "wheel"); //NOTE we rewrite this in useD3Zoom
            }
        }
    });

    onBeforeUnmount(() => {
        hideBrushRectFuncRef.value();
    });

    onUnmounted(() => {
        console.log(`in useD3Brush, in ${debugViewName}, Unmounted!`);

        const gBrush = d3.select(brushRef.value);
        if (gBrush && gBrush.node()) {
            gBrush.remove();
        }
        enableBrushFuncRef.value = () => {};
        disableBrushFuncRef.value = () => {};
        hideBrushRectFuncRef.value = () => {};
    });
    return {
        enableBrushFuncRef,
        disableBrushFuncRef,
        hideBrushRectFuncRef,
    };
};
