import {
    createRouter,
    createWebHistory,
    type RouteLocationNormalized,
} from "vue-router";
import { useMyStore } from "../stores/store";
import { baseUrl } from "../api/api";

const verifyDataset = async (
    to: RouteLocationNormalized,
    from: RouteLocationNormalized
) => {
    // const datasetStore = useDatasetStore();
    // const datasetList = datasetStore.datasetList;
    // console.log(to.path.substring(1));
    // if (!datasetList.some((d) => d === to.path.substring(1))) {
    //     router.push("/404");
    //     return false;
    // }
    const timer = setTimeout(() => {
        console.warn("time out!");
        router.push("/404");
        return false;
    }, 10_000);
    const list = Object.hasOwn(to.params, "datasetName1")
        ? [to.params.datasetName1, to.params.datasetName2]
        : [to.params.datasetName];
    const res = await Promise.all(list.map((d) => fetch(baseUrl + d)));

    // console.log(res);

    if (res.every((d) => d.ok)) {
        clearTimeout(timer);
        return true;
    } else {
        clearTimeout(timer);
        console.warn("route rejected!");
        router.push("/404");
        return false;
    }
};
const router = createRouter({
    history: createWebHistory(import.meta.env.BASE_URL),
    routes: [
        {
            path: "/home",
            redirect: {
                name: "Home",
            },
        },
        {
            path: "/",
            name: "Home",
            component: () => import("../layouts/homeLayout.vue"),
            // component: HomeView,
        },

        {
            path: "/single/:datasetName",
            name: "Dataset",
            // route level code-splitting
            // this generates a separate chunk (About.[hash].js) for this route
            // which is lazy-loaded when the route is visited.
            component: () => import("../layouts/singleLayout.vue"),
            // props: true,
            beforeEnter: [verifyDataset],
        },
        { path: "/:datasetName", redirect: { name: "Dataset" } },
        {
            path: "/compare/:datasetName1/:datasetName2",
            name: "Compare",
            component: () => import("../layouts/compareLayout.vue"),
            beforeEnter: [verifyDataset],
        },
        {
            path: "/404",
            name: "NotFound",
            component: () => import("../layouts/notFoundLayout.vue"),
        },
        { path: "/:pathMatch(.*)", redirect: { name: "NotFound" } },
    ],
});
router.beforeEach((to, from) => {
    const myStore = useMyStore();
    if (to.path !== "/404") {
        myStore.setRouteLoading();
    }
});
router.afterEach((to, from) => {
    const myStore = useMyStore();
    if (to.path !== "/404") {
        myStore.clearRouteLoading();
    }
});
export default router;
