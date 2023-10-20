import { defineStore } from "pinia";
import { computed, ref, shallowRef, watch } from "vue";
import type { Ref } from "vue";
import { ElLoading } from "element-plus";
import type { LoadingInstance } from "element-plus/es/components/loading/src/loading";
import { useWebWorkerFn } from "@/utils/myWebWorker";
import {
    filterEdgeAndComputeDict,
    computeNeighborMasks,
    calcHexbinClusters,
    calcAggregatedLinks,
    calcRank3,
    calcPolar,
    isEmptyDict,
    filterEdgesAndComputeDictInMultiGraph,
} from "@/utils/graphUtils";
import type {
    Type_NodeId,
    Type_RankEmbDiffAlgos,
    Type_polarEmbDiffAlgos,
    Type_polarTopoDistAlgos,
    Type_DashboardsLayoutMode,
    Type_ClearSelMode,
    Type_WhenToRescaleMode,
    Type_SettingsMenuTriggerMode,
    Type_NodesSelectionEntryId,
    SingleDashboard,
    CompDashboard,
    View,
    Dataset,
    RecentDb,
    UrlData,
    Type_TaskTypes,
    Type_DashboardType,
    FetchedGraphData,
    NodeClassificationPredData,
    LinkPredictionPredData,
    GraphClassificationPredData,
    TsneCoord,
    RankView,
    PolarView,
    AggregatedView,
    LinkableView,
    Link,
    NodeCoord,
    DenseView,
    Type_SymbolName,
    Type_GraphId,
    Type_ViewName,
    Type_LabelTypes,
} from "@/types/types";
import {
    isTypeNodeClassification,
    isTypeLinkPrediction,
    isTypeGraphClassification,
} from "@/types/typeFuncs";
import * as d3 from "d3";
import dom2pic from "dom2pic";
import {
    circlePath,
    leftHalfCirclePath,
    rightHalfCirclePath,
    trianglePath,
    leftHalfTrianglePath,
    rightHalfTrianglePath,
    rectPath,
    leftHalfRectPath,
    rightHalfRectPath,
    diamondPath,
    leftHalfDiamondPath,
    rightHalfDiamondPath,
    wyePath,
    leftHalfWyePath,
    rightHalfWyePath,
    starPath,
    leftHalfStarPath,
    rightHalfStarPath,
    eksPath,
    leftHalfEksPath,
    rightHalfEksPath,
} from "@/utils/otherUtils";

export const useMyStore = defineStore("my", () => {
    //dataset
    const datasetList = ref<Array<Dataset>>([]); //normally, no more than 2

    /**
     * hops  symbols
     */
    const symbolHops = ref([
        d3.symbolTriangle2, //0 index but actually 1-step neighbor
        d3.symbolSquare2,
        d3.symbolDiamond,
        d3.symbolWye,
        d3.symbolStar,
    ]); //REVIEW 暂定：统一使用stroke风格的symbol。
    const symbolPathByHopAndParentIndex = ref([
        [leftHalfTrianglePath, rightHalfTrianglePath, trianglePath], //0 index but actually 1-step neighbor
        [leftHalfRectPath, rightHalfRectPath, rectPath],
        [diamondPath, leftHalfDiamondPath, rightHalfDiamondPath],
        [wyePath, leftHalfWyePath, rightHalfWyePath],
        [starPath, leftHalfStarPath, rightHalfStarPath],
        //
        //
        //
        //
        //
        [eksPath, leftHalfEksPath, rightHalfEksPath], //NOTE unreachable
        [leftHalfCirclePath, rightHalfCirclePath, circlePath], //NOTE -1
    ]);
    /**
     * hop symbol names, with respect to `symbolHops`\
     * the index must be consistent!
     */
    const symbolNames = ref<Type_SymbolName[]>([
        "triangle",
        "square",
        "diamond",
        "wye",
        "star",
    ]);
    /**
     * default hops.\
     * should <= `symbolHops.length`
     */
    const defaultHops = ref(2);

    // global ui setting
    const dashboardsLayoutMode = ref<Type_DashboardsLayoutMode>("replace");
    const settingsMenuTriggerMode = ref<Type_SettingsMenuTriggerMode>("click"); //那些setting齿轮，是悬浮触发还是点击触发
    const globalClearSelMode = ref<Type_ClearSelMode>("manual"); //是否手动清除，即每次select接着选还是重新选
    const globalWhenToResizeMode =
        ref<Type_WhenToRescaleMode>("simultaneously");
    const globalLabelType = ref<Type_LabelTypes>("true");

    // algo default
    const defaultRankEmbDiffAlgo = ref<Type_RankEmbDiffAlgos>("center");
    const defaultPolarEmbDiffAlgo = ref<Type_polarEmbDiffAlgos>("center");
    const defaultPolarTopoDistAlgo = ref<Type_polarTopoDistAlgos>("jaccard");
    const defaultCompViewNames = ref<Record<Type_TaskTypes, Type_ViewName[]>>(
        //每一个db的默认views的名称有哪些
        {
            "node-classification": [
                "Topology Space",
                "Latent Space - Model 1",
                "Latent Space - Model 2",
                "Topo + Latent Density - Model 1",
                "Topo + Latent Density - Model 2",
                "Comparative Rank View",
                "Comparative Polar View",
                "Prediction Space - Model 1",
                "Prediction Space - Model 2",
            ],
            "link-prediction": [
                "Topology Space",
                "Latent Space - Model 1",
                "Latent Space - Model 2",
                "Topo + Latent Density - Model 1",
                "Topo + Latent Density - Model 2",
                "Comparative Rank View",
                "Comparative Polar View",
                "Prediction Space - Model 1",
                "Prediction Space - Model 2",
            ],
            "graph-classification": [
                "Topology Space",
                "Latent Space - Model 1",
                "Latent Space - Model 2",
                "Topo + Latent Density - Model 1",
                "Topo + Latent Density - Model 2",
                "Graph Latent Space - Model 1",
                "Graph Latent Space - Model 2",
                "Comparative Rank View",
                "Comparative Polar View",
                "Prediction Space - Model 1",
                "Prediction Space - Model 2",
            ],
        }
    );
    const defaultSingleViewNames = ref<Record<Type_TaskTypes, Type_ViewName[]>>(
        {
            "node-classification": [
                "Topology Space",
                "Latent Space",
                "Topo + Latent Density",
                "Prediction Space",
                // "Feature Space",//编程式动态添加，因为不确定是dense or sparse
            ],
            "link-prediction": [
                "Topology Space",
                "Topo + Latent Density",
                "Latent Space",
                "Prediction Space",
                // "Feature Space",//编程式动态添加，因为不确定是dense or sparse
            ],
            "graph-classification": [
                "Topology Space",
                "Latent Space",
                "Topo + Latent Density",
                "Graph Latent Space",
                "Prediction Space",
                // "Feature Space",
                // "Graph Feature Space",//编程式动态添加，因为不确定是dense or sparse
            ],
        }
    );

    /**
     * Dashboard中默认的view和nodesSelEntries的映射方式\
     * 全局共用，不同的db（不管single还是comp）、不同的view，按viewName去“get”或“set”即可
     */
    const defaultNodesSelectionEntryMapper: Ref<
        Record<
            string,
            {
                source: Array<Type_NodesSelectionEntryId>;
                target: Array<Type_NodesSelectionEntryId>;
            }
        >
    > = ref({
        "Topology Space": {
            source: ["full"],
            target: ["public", "comparative"],
        },
        "Latent Space": {
            source: ["full"],
            target: ["public"],
        },
        "Latent Space - Model 1": {
            source: ["full"],
            target: ["public", "comparative"],
        },
        "Latent Space - Model 2": {
            source: ["full"],
            target: ["public", "comparative"],
        },
        "Topo + Latent Density": {
            source: ["public"],
            target: ["densityOut"],
        },
        "Topo + Latent Density - Model 1": {
            source: ["public"],
            target: ["densityOut"],
        },
        "Topo + Latent Density - Model 2": {
            source: ["public"],
            target: ["densityOut"],
        },
        "Graph Latent Space": {
            source: ["full"],
            target: ["public"],
        },
        "Graph Latent Space - Model 1": {
            source: ["full"],
            target: ["public", "comparative"],
        },
        "Graph Latent Space - Model 2": {
            source: ["full"],
            target: ["public", "comparative"],
        },
        "Comparative Rank View": {
            source: ["comparative", "comparativeSingle"],
            target: ["comparativeOut", "rankOut"],
        },
        "Comparative Polar View": {
            source: ["comparative", "comparativeSingle"],
            target: ["comparativeOut"],
        },
        "Feature Space": {
            source: ["full"],
            target: ["public", "comparative"],
        },
        "Graph Feature Space": {
            source: ["full"],
            target: ["public", "comparative"],
        },
        "Prediction Space": {
            source: ["full"],
            target: ["public"],
        },
        "Prediction Space - Model 1": {
            source: ["full"],
            target: ["public", "comparative"],
        },
        "Prediction Space - Model 2": {
            source: ["full"],
            target: ["public", "comparative"],
        },
    });
    const getViewSourceNodesSelectionEntry = (
        viewName: string
    ): Type_NodesSelectionEntryId[] => {
        return defaultNodesSelectionEntryMapper.value[viewName].source;
        //REVIEW: other cases?
    };
    const getViewTargetNodesSelectionEntry = (
        viewName: string
    ): Type_NodesSelectionEntryId[] => {
        return defaultNodesSelectionEntryMapper.value[viewName].target;
        //REVIEW: other cases?
    };

    /**
     * 默认的entry（去重）及其描述 \
     * NOTE 实际上我们只需要一个全局的就好了。需要的去get、set
     */
    const defaultNodesSelectionEntryDescriptions: Ref<
        Record<Type_NodesSelectionEntryId, string>
    > = ref({
        full: "the full set of graph nodes, usually the source for initial generated views like topo, latent, pred, feat, .et al, to give all views a  unified data source and target paradigm",
        public: "first public selection from 4 spaces to other views",
        comparative: "selections from 4 spaces to comparative views",
        comparativeSingle:
            "a single node selection from 4 spaces to comparative views",
        comparativeOut: "selections from comparative views to other views",
        rankOut:
            "selections from rank view, which may have independent purpose",
        densityOut: "selections from the topo-latent density view",
    });
    const testWatcher = watch(
        defaultNodesSelectionEntryDescriptions,
        (newV) => {
            console.warn("in store, watch called, newV", newV);
        },
        {
            deep: true,
            onTrigger(event) {
                console.warn("in store, watch triggered, event", event);
            },
        }
    );

    //dataset
    /**
     * 获取Dataset的原始数据。
     * @param baseUrl
     * @param datasetName
     * @param storeGraph 是否在本次获取图数据，因为comparative情况下两个model只获取一次就够了。
     * @param storeTrueLabels 是否在本次获取trueLabels，因为comparative情况下两个model只获取一次就够了。
     * @returns
     */
    const fetchOriginDataset = async (
        baseUrl: string,
        datasetName: string,
        storeGraph = true,
        storeTrueLabels = true
    ): Promise<Partial<Dataset>> => {
        const ret: Partial<Dataset> = {};
        const predRes = (await d3.json(
            baseUrl + datasetName + "/prediction-results.json"
        )) as
            | NodeClassificationPredData
            | LinkPredictionPredData
            | GraphClassificationPredData;

        if (isTypeNodeClassification(predRes)) {
            ret.predLabels = predRes.predLabels;

            ret.trueLabels = predRes.trueLabels;

            ret.numNodeClasses =
                predRes.numNodeClasses || new Set(ret.trueLabels).size;

            ret.colorScale = d3
                .scaleOrdinal<string, string>()
                .domain(d3.range(ret.numNodeClasses).map(String))
                .range(d3.schemeCategory10);
        } else if (isTypeLinkPrediction(predRes)) {
            ret.trueAllowEdges = predRes.trueAllowEdges;
            ret.falseAllowEdges = predRes.falseAllowEdges;
            ret.trueUnseenEdgesSorted = predRes.trueUnseenEdgesSorted;
            ret.trueUnseenTopK =
                predRes.trueUnseenTopK || predRes.trueUnseenEdgesSorted
                    ? predRes.trueUnseenEdgesSorted[0].length || 1
                    : 1;

            // if we have trueLabels for nodes
            let trueLabels: string[] = [];
            try {
                const labelTxtRet = await d3.text(
                    baseUrl + datasetName + "/true-labels.txt"
                );
                trueLabels = labelTxtRet.split("\n");

                ret.trueLabels = trueLabels.at(-1)?.match(/^\s+/g)
                    ? trueLabels.slice(0, -1).map(Number)
                    : trueLabels.map(Number);
            } catch (e) {
                console.warn(
                    "in fetchOriginDataset, in link pred, try to get node labelTxtRet, got error",
                    e
                );
            }
        } else if (isTypeGraphClassification(predRes)) {
            // if we have trueLabels for nodes
            let trueLabels: string[] = [];
            try {
                const labelTxtRet = await d3.text(
                    baseUrl + datasetName + "/true-labels.txt"
                );
                trueLabels = labelTxtRet.split("\n");

                ret.trueLabels = trueLabels.at(-1)?.match(/^\s+/g)
                    ? trueLabels.slice(0, -1).map(Number)
                    : trueLabels.map(Number);
            } catch (e) {
                console.warn(
                    "in fetchOriginDataset, in graph classification, try to get node labelTxtRet, got error",
                    e
                );
            }

            ret.graphPredLabels =
                predRes.predLabels ||
                (predRes["pred-labels"] as number[]) ||
                (predRes["pred_labels"] as number[]) ||
                (predRes["predictLabels"] as number[]) ||
                (predRes["predict_labels"] as number[]) ||
                (predRes["predict-labels"] as number[]) ||
                [];
            ret.graphTrueLabels = //two dataset may both be shuffled, resulting in inconsistent orders
                //thus we store both
                predRes.trueLabels ||
                (predRes["true-labels"] as number[]) ||
                (predRes["true_labels"] as number[]) ||
                [];
            ret.numGraphClasses =
                predRes.numGraphClasses ||
                (predRes["num-graph-classes"] as number) ||
                (predRes["num_graph_classes"] as number) ||
                new Set(ret.graphTrueLabels).size;
            ret.graphColorScale = d3
                .scaleOrdinal<string, string>()
                .domain(d3.range(ret.numGraphClasses).map(String))
                .range(d3.schemeCategory10);
            ret.graphIndex =
                predRes.graphIndex ||
                (predRes["graph-index"] as number[]) ||
                (predRes["graph_index"] as number[]);
            // console.log(
            //     "in fetchOriginDataset, in graph-classification, in processing prediction-result.json, graphIndex is",
            //     ret.graphIndex
            // );
            if (ret.graphIndex) ret.graphIndex = ret.graphIndex.map(String);

            ret.phase = predRes.phase as number[];
            if (ret.phase) {
                if (predRes.phaseDict) {
                    ret.phase = predRes.phase as number[];
                    ret.phaseDict =
                        predRes.phaseDict ||
                        predRes["phase-dict"] ||
                        predRes["phase_dict"];
                } else {
                    const nonDupPhaseArr = Array.from(
                        new Set(predRes.phase as string[])
                    );

                    ret.phaseDict = nonDupPhaseArr.reduce(
                        (acc, cur, curI) => ({
                            ...acc,
                            [curI]: cur,
                        }),
                        {}
                    );
                    const invertedDict = nonDupPhaseArr.reduce(
                        (acc, cur, curI) => ({
                            ...acc,
                            [cur]: curI,
                        }),
                        {}
                    ) as Record<string, number>;
                    ret.phase = (predRes.phase as string[]).map(
                        (d) => invertedDict[d]
                    );
                }
            }

            //////////////////////////////////////////////////////////
            ///// SECTION load graph emb
            let embGraph;
            try {
                embGraph = d3.csvParseRows(
                    await d3.text(
                        baseUrl + datasetName + "/graph-embeddings.csv"
                    ),
                    (d: Array<string>) => d.map(Number)
                );
            } catch (e) {
                console.warn(
                    "in fetchOriginDataset, in load embGraph, got error",
                    e,
                    "the result will be",
                    embGraph
                );
            }
            ret.embGraph = embGraph;
            ///// !SECTION load graph emb
            //////////////////////////////////////////////////////////

            ////////////////////////////////////////////////////////////////
            ////// SECTION if we have already calculated graph emb dim-reduction rets
            let graphTsneRet = undefined;
            const graphTsneRetFileNames = [
                "graph-embeddings-tsne-ret.csv",
                "graph-embeddings-tsne-result.csv",
                "graph-embeddings-result.csv",
                "graph-embeddings-ret.csv",
                "graph-embeddings-tsne.csv",
            ];
            for (let i = 0; i < graphTsneRetFileNames.length; ++i) {
                try {
                    graphTsneRet = d3.csvParseRows(
                        await d3.text(
                            baseUrl +
                                datasetName +
                                "/" +
                                graphTsneRetFileNames[i]
                        ),
                        (rawRow: Array<string>, index: number): TsneCoord => ({
                            id: index + "",
                            x: (
                                rawRow.map((s) => Number(s)) as [number, number]
                            )[0],
                            y: (
                                rawRow.map((s) => Number(s)) as [number, number]
                            )[1],
                        })
                    );

                    if (graphTsneRet) break;
                } catch (e) {
                    if (i < graphTsneRetFileNames.length - 1) {
                        console.warn(
                            "in fetchOriginData, dataset name:",
                            datasetName,
                            "try to get graphTsneRet, got error",
                            e,
                            "\nnow try next"
                        );
                        continue;
                    } else {
                        console.warn(
                            "in fetchOriginData, dataset name:",
                            datasetName,
                            "try to get graphTsneRet, got error",
                            e,
                            "finally ret will be",
                            graphTsneRet
                        );
                    }
                }
            }
            ret.graphTsneRet = graphTsneRet;
            ////// !SECTION load already calculated graph emb dim-reduction rets
            ////////////////////////////////////////////////////////////////

            ret.graphUserDefinedFeatureRecord = {}; //self defined feature initial
        } else {
            throw new Error(`undefined task type`);
        }
        ret.taskType = predRes.taskType;

        if (storeGraph) {
            const graphData: FetchedGraphData = (await d3.json(
                baseUrl + datasetName + "/graph.json"
            )) as FetchedGraphData;
            console.log(" in fetchOriginData,graphData", graphData);

            //NOTE graphData.graphs 可能是空的{}
            for (const gid in graphData.graphs) {
                // compatible with both Record<number,{}> and Array<{}>

                // transform multi graphs to flat nodesArr and edgeArr
                const g = graphData.graphs[gid];
                for (const nid of g.nodes) {
                    graphData.nodes[+nid].gid = g.id + "";
                    // const n = graphData.nodes.find((n) => n.id == nid);
                    // if (n) n.gid = g.id;
                }
                for (const eid of g.edges) {
                    graphData.edges[+eid].gid = g.id + "";
                    // const e = graphData.edges.find((e) => e.eid == eid);
                    // if (e) e.gid = g.id;
                }
            }
            ret.numGraphs = !isEmptyDict(graphData.graphs)
                ? Object.keys(graphData.graphs).length
                : 1; //when graphData.graphs={}

            ret.nodes = !isEmptyDict(graphData.graphs)
                ? graphData.nodes.map((d) => ({ ...d, id: d.id + "" })) //ANCHOR DEFINITION
                : graphData.nodes.map((d) => ({
                      ...d,
                      id: d.id + "",
                      gid: "0",
                  }));
            ret.globalNodesDict = ret.nodes.reduce(
                (acc, cur) => ({
                    ...acc,
                    [cur.id]: {
                        gid: cur.gid,
                        hop: -1,
                        parentDbIndex: 0,
                    },
                }),
                {}
            );

            ////////////////////////////////////////////////////////////////
            ////// SECTION node trueLabels for link-pred & graph-classification
            if (
                ret.taskType === "graph-classification" ||
                ret.taskType === "link-prediction"
            ) {
                //NOTE - in link-pred and graph-classify true labels maybe stored in graph.json or true-labels.txt
                if (!ret.trueLabels) {
                    if (Object.hasOwn(graphData.nodes[0], "label")) {
                        ret.trueLabels = graphData.nodes.map((d) => d.label);
                    }
                }
                ret.colorScale = d3
                    .scaleOrdinal<string, string>()
                    .domain(
                        predRes.numNodeClasses
                            ? d3.range(predRes.numNodeClasses).map(String)
                            : ret.trueLabels
                            ? ret.trueLabels.map(String)
                            : []
                    )
                    .range(d3.schemeCategory10);
            }
            ////// !SECTION node trueLabels for link-pred & graph-classification
            ////////////////////////////////////////////////////////////////

            ret.hops = graphData.hops
                ? graphData.hops >= symbolHops.value.length
                    ? symbolHops.value.length
                    : graphData.hops
                : defaultHops.value;

            /////////////////////////////////////////
            ////// SECTION calc filtered edges and edge dict
            if (ret.taskType !== "graph-classification") {
                const {
                    workerFn: edgeWorkerFn,
                    // workerStatus: edgeWorkerStatus,
                    workerTerminate: edgeWorkerTerminate,
                } = useWebWorkerFn(filterEdgeAndComputeDict, {
                    timeout: 20_000,
                    dependencies: [],
                });
                const { edges, nodeMapLink } = await edgeWorkerFn(
                    graphData.nodes.length,
                    graphData.edges
                );
                console.log(
                    "in calc Edge, useWebWorkerFn, got ret: edges.len",
                    edges,

                    "nodeMapLink.len",
                    nodeMapLink
                );
                edgeWorkerTerminate();
                ret.links = edges.map((e) => ({ ...e, gid: "0" }));
                ret.nodeMapLink = nodeMapLink; //ANCHOR - DEFINITION related
            } else {
                if (!ret.graphIndex) {
                    //如果前面没有获取到。用0-n生成
                    ret.graphIndex = Array.from(
                        { length: ret.numGraphs },
                        (v, k) => k + ""
                    );
                }

                const {
                    workerFn: edgeWorkerFn,
                    // workerStatus: edgeWorkerStatus,
                    workerTerminate: edgeWorkerTerminate,
                } = useWebWorkerFn(filterEdgesAndComputeDictInMultiGraph, {
                    timeout: 20_000,
                    dependencies: [],
                });
                const {
                    filteredGraphRecord,
                    filteredGraphArr,
                    filteredEdges,
                    nodeMapLink,
                } = await edgeWorkerFn(
                    graphData.graphs,
                    graphData.nodes.length,
                    graphData.edges.reduce(
                        (acc, cur) => ({ ...acc, [cur.eid]: cur }),
                        {}
                    )
                );
                console.log(
                    "in calc Edge, useWebWorkerFn, got ret: filteredGraphRecord",
                    filteredGraphRecord,
                    "\nfilteredGraphArr",
                    filteredGraphArr,
                    "\nfilteredEdges",
                    filteredEdges,
                    "\nnodeMapLink",
                    nodeMapLink
                );
                edgeWorkerTerminate();
                ret.links = filteredEdges;
                ret.nodeMapLink = nodeMapLink; //ANCHOR - DEFINITION related
                ret.graphRecords = filteredGraphRecord; //这里不考虑when graphData.graphs={}
                ret.graphArr = filteredGraphArr; //这里不考虑when graphData.graphs={}
            }
            ////// !SECTION calc filtered edges and edge dict
            /////////////////////////////////////////

            ret.globalLinksDict = ret.links.reduce(
                (acc, cur) => ({ ...acc, [cur.eid]: cur.gid }),
                {}
            );
            ret.globalLinksRichDict = ret.links.reduce(
                (acc, cur) => ({
                    ...acc,
                    [cur.eid]: {
                        ...cur,
                        gid: cur.gid,
                        source: cur.source,
                        target: cur.target,
                    },
                }),
                {}
            );

            // const graphRecords: Record<
            //     Type_GraphId,
            //     {
            //         nodesRecord: Record<Type_NodeId, Type_GraphId>;
            //         linksRecord: Record<Type_LinkId, Type_GraphId>;
            //     }
            // > = {};
            // const graphArr: Array<{
            //     gid: Type_GraphId;
            //     nodes: Array<Type_NodeId>;
            //     links: Array<Type_LinkId>;
            // }> = [];
            // for (const gid in graphData.graphs) {
            //     // compatible with both Record<number,{}> and Array<{}>

            //     const g = graphData.graphs[gid];
            //     const filteredLinks = g.edges.filter(
            //         (eid) => ret.globalLinksDict![eid]
            //     );
            //     graphArr.push({
            //         gid: g.id,
            //         nodes: g.nodes.map(String),
            //         links: filteredLinks.map(String),
            //     });
            //     // transform nodesArr in each graph into nodesRecord, so as edges
            //     graphRecords[gid + ""] = {
            //         nodesRecord: graphData.graphs[gid].nodes.reduce(
            //             (acc, cur) => ({ ...acc, [cur + ""]: gid + "" }),
            //             {}
            //         ),
            //         linksRecord: filteredLinks.reduce(
            //             (acc, cur) => ({ ...acc, [cur + ""]: gid + "" }),
            //             {}
            //         ),
            //     };
            // }

            // ret.graphRecords = !isEmptyDict(graphData.graphs)
            //     ? graphRecords
            //     : {
            //           "0": {
            //               nodesRecord: graphData.nodes.reduce(
            //                   (acc, cur) => ({ ...acc, [cur.id + ""]: "0" }),
            //                   {}
            //               ),
            //               linksRecord: graphData.nodes.reduce(
            //                   (acc, cur) => ({ ...acc, [cur.id + ""]: "0" }),
            //                   {}
            //               ),
            //           },
            //       }; //when graphData.graphs={}

            /////////////////////////////////////////
            ////// SECTION calc neighbor masks
            const {
                workerFn: neighborMasksWorkerFn,
                // workerStatus: neighborMasksWorkerStatus,
                workerTerminate: neighborMasksWorkerTerminate,
            } = useWebWorkerFn(computeNeighborMasks, {
                timeout: 30_000,
                dependencies: ["http://localhost:5173/workers/bitset.js"],
            });
            const {
                neighborMasks,
                neighborMasksByHop,
                neighborMasksByHopPure,
            } = await neighborMasksWorkerFn(
                graphData.nodes.length, //raw
                ret.nodeMapLink, //raw
                graphData.hops || defaultHops.value //raw
            );
            console.log(
                "in calc NeighborMask, useWebWorkerFn, got ret: neighborMasks.length",
                neighborMasks.length,
                "\nneighborMasksByHop.len",
                neighborMasksByHop.length,
                "\nneighborMasksByHopPure.len",
                neighborMasksByHopPure.length,
                "\nneighborMasksByHop",
                neighborMasksByHop,
                "\nneighborMasksByHopPure",
                neighborMasksByHopPure
            );
            neighborMasksWorkerTerminate();
            ////// !SECTION calc neighbor masks
            /////////////////////////////////////////

            ret.neighborMasks = neighborMasks; //ANCHOR - DEFINITION related
            ret.neighborMasksByHop = neighborMasksByHop; //ANCHOR - DEFINITION related
            ret.neighborMasksByHopPure = neighborMasksByHopPure; //ANCHOR - DEFINITION related

            ////////////////////////////////////////////////////////////////
            ////// SECTION if we have pre-calculated force-directed graph layout
            let graphCoordsRet = undefined;
            const layoutFileNames = [
                "graph-layout.json",
                "initial-layout.json",
            ];
            for (let i = 0; i < layoutFileNames.length; ++i) {
                try {
                    const jsonRet = await d3.json<any>(
                        baseUrl + datasetName + "/" + layoutFileNames[i]
                    );
                    graphCoordsRet = Object.hasOwn(
                        jsonRet,
                        "forceDirectedLayout"
                    )
                        ? Object.hasOwn(jsonRet.forceDirectedLayout, "nodes")
                            ? jsonRet.forceDirectedLayout.nodes
                            : jsonRet.forceDirectedLayout
                        : Object.hasOwn(jsonRet, "nodes")
                        ? jsonRet.nodes
                        : jsonRet;
                    if (graphCoordsRet) {
                        graphCoordsRet = graphCoordsRet.map((d) => ({
                            ...d,
                            id: d.id + "", //NOTE maybe number
                        }));
                        break;
                    }
                } catch (e) {
                    if (i < layoutFileNames.length - 1) {
                        console.warn(
                            "in fetchOriginData, dataset name:",
                            datasetName,
                            "try to get graphCoordsRet, got error:",
                            e,
                            "now try next"
                        );
                        continue;
                    } else {
                        console.warn(
                            "in fetchOriginData, dataset name:",
                            datasetName,
                            "try to get graphCoordsRet, got error:",
                            e,
                            "finally ret will be",
                            graphCoordsRet
                        );
                    }
                } finally {
                    ret.graphCoordsRet = graphCoordsRet;
                }
            }
            ////// !SECTION load pre-calculated force-directed graph layout
            ////////////////////////////////////////////////////////////////

            ////////////////////////////////////////////////////////
            /// SECTION load node sparse features
            let sparseFeaturesRet;
            try {
                sparseFeaturesRet = await d3.json<any>(
                    baseUrl + datasetName + "/" + "node-sparse-features.json"
                );
                ret.nodeSparseFeatureValues =
                    sparseFeaturesRet.nodeFeatureValues;
                ret.nodeSparseFeatureIndexes =
                    sparseFeaturesRet.nodeFeatureIndexes;
                ret.nodeSparseFeatureIndexToSemantics =
                    sparseFeaturesRet.featureIndexToWord;
                ret.numNodeSparseFeatureDims =
                    sparseFeaturesRet.numNodeFeatureDims ||
                    Object.keys(ret.nodeSparseFeatureIndexToSemantics || {})
                        .length ||
                    0;
            } catch (e) {
                console.warn(
                    "in fetchOriginDataset, try to get node features, error:",
                    e,
                    "\nresults will be:",
                    "\nret.nodeSparseFeatureValues",
                    ret.nodeSparseFeatureValues,
                    "\nret.nodeSparseFeatureValues",
                    ret.nodeSparseFeatureIndexes,
                    "\nret.nodeSparseFeatureValues",
                    ret.nodeSparseFeatureIndexToSemantics,
                    "\nret.nodeSparseFeatureValues",
                    ret.numNodeSparseFeatureDims
                );
            }
            /// !SECTION load node sparse features
            ////////////////////////////////////////////////////////
        }

        ////////////////////////////////////////////////////////
        /// SECTION load node dense features
        let denseNodeFeatures:
            | d3.DSVParsedArray<{ id: Type_NodeId; [key: string]: unknown }>
            | undefined = undefined;
        try {
            const textRet = await d3.text(
                // baseUrl + datasetName + "/node-features.csv"
                baseUrl + datasetName + "/node-dense-features.csv"
            );
            const rows = d3.csvParseRows(textRet);
            const notHasHeader = rows[0].every((value) =>
                Boolean(Number(value))
            );

            if (notHasHeader) {
                let columns: string[] = [];
                denseNodeFeatures = d3.csvParseRows(
                    textRet,
                    (row: Array<string>, i) => {
                        if (i === 0) {
                            columns = row.map((d, j) => `feat-${j}`);
                        }
                        const ret: {
                            id: Type_NodeId;
                            [key: string]: unknown;
                        } = { id: "" };
                        for (const j in row) {
                            //NOTE index, not value!
                            ret[columns[j]] = Number(row[j]);
                            ret.id = i + "";
                        }
                        return ret;
                    }
                ) as d3.DSVParsedArray<{
                    [key: string]: unknown;
                    id: Type_NodeId;
                }>;
                denseNodeFeatures.columns = columns;
            } else {
                denseNodeFeatures = d3.csvParse<
                    { id: Type_NodeId; [key: string]: unknown },
                    string
                >(
                    await d3.text(
                        baseUrl + datasetName + "/node-dense-features.csv"
                    ),
                    (d, i) => {
                        const ret: { [key: string]: unknown; id: string } = {
                            id: i + "",
                        };
                        for (const k in d) {
                            ret[k] = parseFloat(d[k] as string);
                            d["id"] = i + "";
                        }
                        return ret;
                    }
                );
            }
        } catch (e) {
            console.warn(
                "in fetchOriginDataset, try to get denseNodeFeatures, error:",
                e,
                "\nthe result will be:\ndenseNodeFeatures",
                denseNodeFeatures
            );
        }
        /// !SECTION load node features
        ////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////
        ///// SECTION load emb
        const embNode = d3.csvParseRows(
            await d3.text(baseUrl + datasetName + "/node-embeddings.csv"),
            (d: Array<string>) => d.map(Number)
        );
        ///// !SECTION load emb
        //////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////
        ////// SECTION if we have already calculated emb dim-reduction rets
        let tsneRet = undefined;
        const tsneRetFileNames = [
            "initial-layout.json",
            "node-embeddings-tsne-ret.csv",
            "node-embeddings-ret.csv",
            "node-embeddings-tsne-result.csv",
            "node-embeddings-tsne.csv",
            "node-embeddings-result.csv",
            "node-embeddings-result.csv",
        ];
        for (let i = 0; i < tsneRetFileNames.length; ++i) {
            try {
                if (tsneRetFileNames[i].endsWith(".csv")) {
                    tsneRet = d3.csvParseRows(
                        await d3.text(
                            baseUrl + datasetName + "/" + tsneRetFileNames[i]
                        ),
                        (rawRow: Array<string>, index: number): TsneCoord => ({
                            id: index + "",
                            x: (
                                rawRow.map((s) => Number(s)) as [number, number]
                            )[0],
                            y: (
                                rawRow.map((s) => Number(s)) as [number, number]
                            )[1],
                        })
                    );
                    if (tsneRet) break;
                } else if (tsneRetFileNames[i].endsWith(".json")) {
                    const jsonRet =
                        (await d3.json<any>(
                            baseUrl + datasetName + "/" + tsneRetFileNames[i]
                        )) || {};
                    tsneRet =
                        jsonRet[
                            Object.keys(jsonRet).find((key) =>
                                key.match(
                                    /^(embNode|emb_node|nodeEmb|node_emb)/
                                )
                            ) || ""
                        ];
                    if (tsneRet) {
                        tsneRet = tsneRet.map(
                            (d: [number, number], index: number) => ({
                                id: index + "",
                                x: d[0],
                                y: d[1],
                            })
                        );
                        break;
                    }
                }
            } catch (e) {
                if (i < tsneRetFileNames.length - 1) {
                    console.warn(
                        "in fetchOriginData, dataset name:",
                        datasetName,
                        "try to get tsneRet, got error",
                        e,
                        "\nnow try next"
                    );
                    continue;
                } else {
                    console.warn(
                        "in fetchOriginData, dataset name:",
                        datasetName,
                        "try to get tsneRet, got error",
                        e,
                        "finally ret will be",
                        tsneRet
                    );
                }
            }
        }
        ////// !SECTION load already calculated emb dim-reduction rets
        ////////////////////////////////////////////////////////////////

        //ANCHOR DEFINITION related
        return {
            ...ret,
            embNode,
            tsneRet,
            denseNodeFeatures,
            name: datasetName,
        };
    };

    /**
     * 由名字获取dataset
     */
    const getDatasetByName = (datasetName: string) => {
        const ret = datasetList.value.find((ds) => ds.name === datasetName);
        if (ret) return ret;
        else throw new Error("Couldn't find Dataset by name: " + datasetName);
    };
    const addDataset = (datasetObj: Dataset) => {
        let nameTmp = datasetObj.name;
        let i = 1; //重名的就重命名
        while (datasetList.value.find((d) => d.name === nameTmp)) {
            nameTmp = `${datasetObj.name} (${i})`;
            i++;
        }

        datasetList.value.push({ ...datasetObj, name: nameTmp });
    };
    const removeDataset = (datasetName: string) => {
        datasetList.value = datasetList.value.filter(
            (ds) => ds.name !== datasetName
        );
    };

    // dashboard
    const singleDashboardList = ref<Array<SingleDashboard>>([]);
    const recentSingleDashboardList = ref<Array<RecentDb>>([]);
    // const representedSingleDashboardList = ref<Array<string>>([]);
    const recentLen = ref(3);
    const maxRecentLen = ref(7);
    const minRecentLen = ref(2);
    const getSingleDashboardById = (id: string) => {
        const ret = singleDashboardList.value.find((db) => db.id === id);
        if (ret) return ret;
        else throw new Error("Couldn't find Single Dashboard by id: " + id);
    };

    const renderableRecentSingleDashboards = computed(() =>
        //每次插入或者替换，都会出发renderIndex的复制，然后将归隐的renderIndex清0，
        //这样如果意外扩充renderList长度，能保证排序的时候总在前面。
        //当renderIndex相同，按索引排（最近访问次序）
        recentSingleDashboardList.value
            // .filter((d) => !d.isRepresented)
            .filter((d) => !getSingleDashboardById(d.id)!.isRepresented)
            .slice(-recentLen.value)
            .sort((a, b) =>
                a.renderIndex != b.renderIndex
                    ? a.renderIndex - b.renderIndex
                    : recentSingleDashboardList.value.indexOf(a) -
                      recentSingleDashboardList.value.indexOf(b)
            )
    );
    const toSingleDashboardById = (dbId: string) => {
        //we assume dbId is in recentList by default
        const i = recentSingleDashboardList.value.findIndex(
            (d) => d.id === dbId
        );
        if (i < 0) {
            throw new Error(`to singleDashboard id '${dbId}' not found!`);
        } else {
            if (i < recentSingleDashboardList.value.length - recentLen.value) {
                //说明renderList里面没有，应当更新renderIndex
                recentSingleDashboardList.value[i].renderIndex =
                    recentSingleDashboardList.value.at(
                        -recentLen.value
                    )!.renderIndex;
                recentSingleDashboardList.value.at(
                    -recentLen.value
                )!.renderIndex = 0;
            }
            const to = recentSingleDashboardList.value.splice(i, 1);
            // if (to[0].isRepresented) to[0].isRepresented = false;
            recentSingleDashboardList.value.push(to[0]); //这里是个数组
        }
    };

    const addSingleDashboard = (
        dbObj: Partial<SingleDashboard>,
        viewsDefinitionList: string[]
    ) => {
        //name
        const nameTmp = dbObj.refDatasetName!; //
        console.log("in add dashboard, nameTmp", nameTmp);
        let nameRet = nameTmp;
        let i = 1; //重名的就重命名
        while (singleDashboardList.value.find((d) => d.name === nameRet)) {
            console.log("in add dashboard, while ", i);
            nameRet = `${nameTmp}-(${i})`;
            i++;
        }
        dbObj.name = nameRet;

        // 这些默认的entry，可以通过一个外部的prescription配方函数来预先定义好
        dbObj.nodesSelections = Array.from(
            new Set(
                viewsDefinitionList.flatMap((name) => {
                    const { source, target } =
                        defaultNodesSelectionEntryMapper.value[name];
                    return [...source, ...target];
                }) //根据仅在这个db中的viewNames，拿到mapper中所有的src、tgt的entryIds，去重
            )
        ).reduce((acc, cur) => ({ ...acc, [cur]: {} }), {});
        //{ id1:{}, id2:{}, id3:{}, ...}

        //compDashboard创建时进行的计算默认为空
        dbObj.calcOnCreatedPromise = new Promise(() => {}); //NOTE  空的Promise
        dbObj.tsneCalcPromise = new Promise(() => {});
        dbObj.tsneCalcPromiseReject = () => {};

        //views definition initial
        dbObj.viewsDefinitionList = viewsDefinitionList.map(
            (d) => initialNewView(d) //NOTE 添加新view到DefinitionList，也会写相同的代码，故可以提取公共
        );

        //UI related effect
        dbObj.clearSelMode = globalClearSelMode.value;
        dbObj.whenToRescaleMode = globalWhenToResizeMode.value;
        dbObj.isHighlightCorrespondingNode = true;
        dbObj.highlightedNodeIds = {};
        dbObj.highlightedGreedyGraphIds = {};
        dbObj.highlightedWholeGraphIds = {};

        dbObj.labelType = globalLabelType.value;

        //类型完成
        singleDashboardList.value.push(dbObj as SingleDashboard); //NOTE 为了这里能跑通，View定义中不应当包含Ref

        //recent related effect
        if (recentSingleDashboardList.value.length < recentLen.value) {
            recentSingleDashboardList.value.push({
                id: dbObj.id!,
                renderIndex: recentSingleDashboardList.value.length,
                // isRepresented: false,
            });
        } else {
            const oldestRender = recentSingleDashboardList.value.at(
                -recentLen.value
            ) as RecentDb;
            recentSingleDashboardList.value.push({
                id: dbObj.id!,
                renderIndex: oldestRender.renderIndex,
                // isRepresented: false,
            });
            oldestRender.renderIndex = 0;
        }
    };

    // compDashboard
    const compDashboardList = ref<Array<CompDashboard>>([]);
    const recentCompDashboardList = ref<Array<RecentDb>>([]);
    const getCompDashboardById = (id: string) => {
        const ret = compDashboardList.value.find((db) => db.id === id);
        if (ret) return ret;
        else
            throw new Error("Couldn't find Comparative Dashboard by id: " + id);
    };
    const renderableRecentCompDashboards = computed(() =>
        //每次插入或者替换，都会出发renderIndex的复制，然后将归隐的renderIndex清0，
        //这样如果意外扩充renderList长度，能保证排序的时候总在前面。
        //当renderIndex相同，按索引排（最近访问次序）
        recentCompDashboardList.value
            .filter((d) => !getCompDashboardById(d.id)!.isRepresented)
            .slice(-recentLen.value)
            .sort((a, b) =>
                a.renderIndex != b.renderIndex
                    ? a.renderIndex - b.renderIndex
                    : recentCompDashboardList.value.indexOf(a) -
                      recentCompDashboardList.value.indexOf(b)
            )
    );
    const toCompDashboardById = (dbId: string) => {
        //we assume dbId is in recentList by default
        const i = recentCompDashboardList.value.findIndex((d) => d.id === dbId);
        if (i < 0) {
            throw new Error(`to compDashboard id '${dbId}' not found!`);
        } else {
            if (i < recentCompDashboardList.value.length - recentLen.value) {
                //说明renderList里面没有，应当更新renderIndex
                recentCompDashboardList.value[i].renderIndex =
                    recentCompDashboardList.value.at(
                        -recentLen.value
                    )!.renderIndex;
                recentCompDashboardList.value.at(
                    -recentLen.value
                )!.renderIndex = 0;
            }
            const to = recentCompDashboardList.value.splice(i, 1);
            recentCompDashboardList.value.push(to[0]); //这里是个数组
        }
    };

    /**
     * 添加ComparativeDb
     * @param dbObj 可能是不完整的对象
     */
    const addCompDashboard = (
        dbObj: Partial<CompDashboard>,
        viewsDefinitionList: string[]
    ) => {
        //name
        const nameTmp =
            dbObj.refDatasetsNames![0] + //确定非空
            " vs. " +
            dbObj?.refDatasetsNames![1];
        let nameRet = nameTmp;
        let i = 1; //重名的就重命名
        while (compDashboardList.value.find((d) => d.name === nameRet)) {
            nameRet = `${nameTmp}-(${i})`;
            i++;
        }
        dbObj.name = nameRet;

        // 这些默认的entry，可以通过一个外部的prescription配方函数来预先定义好
        dbObj.nodesSelections = Array.from(
            new Set(
                viewsDefinitionList.flatMap((name) => {
                    const { source, target } =
                        defaultNodesSelectionEntryMapper.value[name];
                    return [...source, ...target];
                }) //根据仅在这个db中的viewNames，拿到mapper中所有的src、tgt的entryIds，去重
            )
        ).reduce((acc, cur) => ({ ...acc, [cur]: {} }), {});
        //{ id1:{}, id2:{}, id3:{}, ...}

        //compDashboard创建时进行的计算默认为空
        dbObj.calcOnCreatedPromise = new Promise(() => {}); //NOTE  空的Promise
        dbObj.tsne1CalcPromise = new Promise(() => {});
        dbObj.tsne1CalcPromiseReject = () => {};
        dbObj.tsne2CalcPromise = new Promise(() => {});
        dbObj.tsne2CalcPromiseReject = () => {};

        //views definition initial
        dbObj.viewsDefinitionList = viewsDefinitionList.map(
            (d) => initialNewView(d) //NOTE 添加新view到DefinitionList，也会写相同的代码，故可以提取公共
        );

        //UI related effect
        dbObj.clearSelMode = globalClearSelMode.value;
        dbObj.whenToRescaleMode = globalWhenToResizeMode.value;
        dbObj.isHighlightCorrespondingNode = true;
        dbObj.highlightedNodeIds = {};

        dbObj.labelType = globalLabelType.value;

        dbObj.rankWorker = useWebWorkerFn<typeof calcRank3>(calcRank3, {
            //NOTE： 此创建不可重用，因为返回的status是ref，若重用则共享状态，出现混乱。
            timeout: 20_000,
            dependencies: [
                "http://localhost:5173/workers/d3js.org_d3.v7.js",
                // "https://d3js.org/d3.v7.min.js",
                "http://localhost:5173/workers/distance.js", // REVIEW temporary
            ],
        });
        dbObj.polarWorker = useWebWorkerFn<typeof calcPolar>(calcPolar, {
            //NOTE： 此创建不可重用，因为返回的status是ref，若重用则共享状态，出现混乱。
            timeout: 20_000,
            dependencies: [
                "http://localhost:5173/workers/d3js.org_d3.v7.js",
                // "https://d3js.org/d3.v7.min.js",
                "http://localhost:5173/workers/bitset.js",
                "http://localhost:5173/workers/distance.js", // REVIEW temporary
            ],
        });
        //类型完成
        compDashboardList.value.push(dbObj as CompDashboard); //NOTE 为了这里能跑通，View定义中不应当包含Ref

        //recent related effect
        if (recentCompDashboardList.value.length < recentLen.value) {
            recentCompDashboardList.value.push({
                id: dbObj.id!, //确定非空
                renderIndex: recentCompDashboardList.value.length,
                // isRepresented: false,
            });
        } else {
            const oldestRender = recentCompDashboardList.value.at(
                -recentLen.value
            ) as RecentDb; //假定没有undefined
            recentCompDashboardList.value.push({
                id: dbObj.id!, //确定非空
                renderIndex: oldestRender.renderIndex,
                // isRepresented: false,
            });
            oldestRender.renderIndex = 0;
        }
    };

    // view
    /**
     * 定义一个view时，初始化过程
     * @param viewName
     * @returns
     */
    const initialNewView = (viewName: string): View => {
        const obj: View = {
            viewName: viewName,
            headerComp: shallowRef(null),
            bodyComp: shallowRef(null),
            initialWidth: 500,
            initialHeight: 541, //head一般是2.5个em，一般是40px，加上border，这个数字刚刚好

            bodyHeight: 0,
            bodyWidth: 0,
            bodyMargins: { top: 0.05, right: 0.05, bottom: 0.05, left: 0.05 },

            resizeEndSignal: false,
            isBrushEnabled: true,
            brushEnableFunc: () => {}, //NOTE 在view组件挂在之后更新
            brushDisableFunc: () => {}, //NOTE 在view组件挂在之后更新
            panEnableFunc: () => {}, //NOTE 在view组件挂在之后更新
            panDisableFunc: () => {}, //NOTE 在view组件挂在之后更新
            hideRectWhenClearSelFunc: () => {}, //NOTE 在view组件挂在之后更新
            resetZoomFunc: () => {}, //NOTE 在view组件挂在之后更新
            isGettingSnapshot: false,
            gettingSnapShotError: undefined,
            snapshotBase64: "",

            bodyProps: {},
            headerProps: {},

            setAttr(attr, value: any) {
                console.log(
                    "view definition of",
                    this.viewName,
                    ": now setting attr",
                    attr,
                    "before:",
                    this[attr],
                    "new",
                    value
                );
                this[attr] = value;
                return this;
            },
        };

        // ANCHOR prescription
        // if (viewName.match(/topo|latent|pred|feat/i)) {
        //      //
        // }

        // ANCHOR prescription
        if (viewName.match(/latent/i)) {
            (obj as AggregatedView & LinkableView).isShowLinks = false;

            (obj as AggregatedView & LinkableView).isShowAggregation = false;
            (obj as AggregatedView & LinkableView).isShowMesh = false;
            (obj as AggregatedView & LinkableView).meshRadius = 30;
            (obj as AggregatedView & LinkableView).pointCluster = {};
            (obj as AggregatedView & LinkableView).hexbin = undefined;
            (obj as AggregatedView & LinkableView).clusters = [];
            (obj as AggregatedView & LinkableView).aggregatedLinks = [];

            (obj as AggregatedView & LinkableView).sourceCoords = [];
            (obj as AggregatedView & LinkableView).aggregatedCoords = [];
            (obj as AggregatedView & LinkableView).rescaledCoords = [];

            // obj.which = +viewName.charAt(-1);
            //NOTE: 像which，headerComp，这些在compDashboard创建后决定。因为是和compDb创建和运行的具体状态相关的

            obj.initialWidth = 800;
            return obj as AggregatedView & LinkableView;
        }

        // ANCHOR prescription
        if (viewName.match(/rank/i)) {
            obj.initialWidth = 800;
            obj.isBrushEnabled = false; //暗示用户先drag line再brush
            obj.bodyMargins.bottom = 0.1; //可能要多给点
            obj.bodyMargins.left = 0.1; //可能要多给点

            (obj as RankView & AggregatedView).rankEmbDiffAlgo =
                defaultRankEmbDiffAlgo.value; //REVIEW 一上来留空会更好一些？让用户知道哪里是干嘛的
            (obj as RankView & AggregatedView).rankData = [];

            (obj as AggregatedView).isShowAggregation = false;
            (obj as AggregatedView).meshRadius = 30;

            return obj as RankView;
        }

        if (viewName.match(/polar/i)) {
            obj.initialWidth = 800;

            (obj as PolarView).polarEmbDiffAlgo = defaultPolarEmbDiffAlgo.value;
            (obj as PolarView).polarTopoDistAlgo =
                defaultPolarTopoDistAlgo.value;
            (obj as PolarView).polarData = [];

            (obj as AggregatedView).isShowAggregation = false;
            (obj as AggregatedView).meshRadius = 30;
            return obj as PolarView;
        }

        if (viewName.match(/dense/i)) {
            (obj as DenseView).isRelative = false;
        }
        return obj as View;
    };
    const getTypedDashboardById = (
        id: string,
        dbType: Type_DashboardType
    ): SingleDashboard | CompDashboard | undefined => {
        if (dbType === "single") {
            return getSingleDashboardById(id);
        } else {
            return getCompDashboardById(id);
        }
    };
    const getTypeReducedDashboardById = (
        db: CompDashboard | SingleDashboard | string
    ): CompDashboard | SingleDashboard | undefined => {
        if (typeof db === "string") {
            let dbObj;
            //NOTE:这里可能会报错！
            try {
                dbObj = getCompDashboardById(db);
                if (dbObj) return dbObj;
            } catch (e) {
                // if (!dbObj) {
                try {
                    dbObj = getSingleDashboardById(db);
                    if (dbObj) return dbObj;
                } catch (ee) {
                    throw new Error(
                        "Neither single or comparative dashboard found!, param: " +
                            db
                    );
                }
                // }
            }
        } else {
            return db;
        }
    };
    const getViewIndexByName = (
        db: SingleDashboard | CompDashboard | string,
        viewName: string
    ) => {
        if (!db) {
            throw new Error("In getViewIndexByName, undefined db: " + db);
        }
        const dbObj = getTypeReducedDashboardById(db);

        const ret = dbObj!.viewsDefinitionList.findIndex(
            (d) => d.viewName === viewName
        );
        if (ret < 0)
            throw new Error("Couldn't find Index of viewName: " + viewName);
        return ret;
    };
    const getViewByName = (
        db: SingleDashboard | CompDashboard | string,
        viewName: string
    ) => {
        const dbObj = getTypeReducedDashboardById(db);
        const index = getViewIndexByName(dbObj!, viewName);

        if (typeof index === "number") {
            return dbObj?.viewsDefinitionList[index];
        }
        throw new Error(
            "undefined viewName" + viewName + "in db" + db.toString()
        );
    };
    const insertViewAfterName = (
        db: SingleDashboard | CompDashboard | string,
        viewName: string,
        obj: View
    ) => {
        const dbObj = getTypeReducedDashboardById(db);
        const index = getViewIndexByName(dbObj!, viewName);
        if (typeof index == "number") {
            dbObj?.viewsDefinitionList.splice(index + 1, 0, obj);
            return getViewByName(db, obj.viewName);
        }
        return undefined;
    };

    const getPrincipalViewOfDashboard = (
        db: SingleDashboard | CompDashboard | string
    ) => {
        const dbObj = getTypeReducedDashboardById(db)!;
        const { fromViewName } = dbObj;
        return getViewByName(dbObj, fromViewName);
    };

    // snapshot
    const removeSnapshotsOfDashboard = (
        db: SingleDashboard | CompDashboard | string
    ) => {
        const dbObj = getTypeReducedDashboardById(db)!;
        if (dbObj) {
            dbObj.viewsDefinitionList.forEach((view) => {
                view.isGettingSnapshot = false;
                view.gettingSnapShotError = undefined;
                view.snapshotBase64 = "";
            });
        }
    };
    const calcSnapshotsOfDashboard = async (
        db: SingleDashboard | CompDashboard | string
    ) => {
        const dbObj = getTypeReducedDashboardById(db)!;
        dbObj.viewsDefinitionList.forEach((view) => {
            view.isGettingSnapshot = true;
            view.gettingSnapShotError = undefined;
        });

        const viewBodies = document.querySelectorAll(
            `.resizable-box-bodies-${dbObj.id}`
            // ".resizable-box-body"
            // `#${dbObj.id} .resizable-box-body`
        ); //LINK src/components/publicViews/resizableBox.vue
        if (viewBodies) {
            const viewBodiesArr = Array.from(viewBodies);
            const dom2PicArr = viewBodiesArr.map(
                (d) =>
                    new dom2pic({
                        root: d,
                        background: "#fff",
                    })
            );
            await Promise.all(
                dom2PicArr.map(async (d, i) => {
                    try {
                        const pic = await d.toPng();
                        dbObj.viewsDefinitionList[i].isGettingSnapshot = false;
                        if (pic) {
                            dbObj.viewsDefinitionList[i].snapshotBase64 = pic;
                        }
                    } catch (e) {
                        console.log(
                            "in calcSnapshotsOfDashboard, caught error",
                            e
                        );
                        dbObj.viewsDefinitionList[i].gettingSnapShotError = e;
                    }
                })
            );
        }
    };

    /**
     * 计算某1个Dashboard的Snapshot of Principal View。\
     * 注意，和计算"当前Dashboard"的所有snapshots不同的是，"当前Dashboard"一定可以获得当前各view的宽高\
     * 而切换dashboard会使view的尺寸变成0\
     * html的原生<img>仅需style上的一个尺寸，即可按比例缩放图片，因此不需要宽高数据\
     * 但是svg中的<image>需要手动指定尺寸，此时取view的尺寸是不一定安全的\
     * 因此，此计算完成之后，应当将宽高存储在dashboard的父级数据结构中，而不是每个view中。\
     * 或者从base64获得长宽尺寸
     * @param db
     */
    const calcPrincipalSnapshotOfDashboard = async (
        db: SingleDashboard | CompDashboard | string
    ) => {
        const dbObj = getTypeReducedDashboardById(db)!;
        const view = getPrincipalViewOfDashboard(dbObj);
        const { fromViewName } = dbObj;
        const viewIndex = getViewIndexByName(dbObj, fromViewName);

        view!.isGettingSnapshot = true;
        view!.gettingSnapShotError = undefined;
        const viewBodies = document.querySelectorAll(
            `.resizable-box-bodies-${dbObj.id}`
            // ".resizable-box-body"
            // `#${dbObj.id} .resizable-box-body`
        ); //LINK src/components/publicViews/resizableBox.vue

        console.log(
            "in calcPrincipalSnapshotOfDashboard,",
            "dbName",
            dbObj.name,
            "fromViewName",
            fromViewName,
            "\nviewIndex",
            viewIndex,
            "\nviewBodies",
            viewBodies
        );
        if (viewBodies) {
            const viewBodiesArr = Array.from(viewBodies);

            try {
                const dom2Pic = new dom2pic({
                    root: viewBodiesArr[viewIndex],
                    background: "#fff",
                });
                const pic = await dom2Pic.toPng();

                view!.isGettingSnapshot = false;
                if (pic) {
                    view!.snapshotBase64 = pic;
                }
            } catch (e) {
                view!.gettingSnapShotError = e;
            }
        }
    };
    const getPrincipalSnapshotOfDashboard = (
        db: SingleDashboard | CompDashboard | string
    ) => {
        const dbObj = getTypeReducedDashboardById(db)!;
        const { fromViewName } = dbObj;
        const view = getViewByName(dbObj, fromViewName)!;
        return view.snapshotBase64;
    };

    const calcAggregatedViewData = async <T extends NodeCoord>(
        viewDef: AggregatedView,
        coordsGetter: (v: any) => T[], //which coord?
        extentFn = (): [[number, number], [number, number]] => [
            [
                viewDef.bodyWidth * viewDef.bodyMargins.left,
                viewDef.bodyHeight * viewDef.bodyMargins.top,
            ],
            [
                viewDef.bodyWidth * (1 - viewDef.bodyMargins.right),
                viewDef.bodyHeight * (1 - viewDef.bodyMargins.bottom),
            ],
        ],
        labels: number[],
        isCalcAggregatedLinks = false,
        linkArr: Link[]
    ): Promise<void> => {
        const { hb, pointCluster, clusters } = calcHexbinClusters<T>(
            coordsGetter(viewDef),
            extentFn,
            (d: T) => d.x,
            (d: T) => d.y,
            viewDef.meshRadius,
            (d: T) => labels[+d.id]
        );
        console.log(
            "in myStore.calcAggregatedViewData, got pointCluster",
            pointCluster
        );
        viewDef.pointCluster = pointCluster;
        viewDef.hexbin = hb;
        viewDef.clusterRadiusScale = d3
            .scaleRadial()
            .range([10, viewDef.meshRadius])
            .domain(
                d3.extent(clusters.map((d) => d.pointIds.length)) as [
                    number,
                    number
                ]
            );
        viewDef.clusters = clusters;
        viewDef.aggregatedCoords = clusters.map((d) => ({
            id: d.id,
            x: d.pointCenter.x,
            y: d.pointCenter.y,
            mag: d.pointIds.length,
            arcs: d3
                .pie<[string, number]>()
                .value((entry: [string, number]) => entry[1])
                .sort((entryA: [string, number], entryB: [string, number]) =>
                    entryA[0].localeCompare(entryB[0])
                )(Object.entries(d.count)), //count 是键值对，即按某种属性统计，各属性多少个
        }));

        if (isCalcAggregatedLinks) {
            const linkRet = calcAggregatedLinks(
                clusters.map((d) => d.pointIds),
                linkArr,
                (nodeId) => pointCluster[nodeId]
            );
            console.log(
                "in myStore.calcAggregatedViewData, got linkRet",
                linkRet
            );
            viewDef.aggregatedLinks = linkRet;
        }
    };

    // layout fetch & calc data on Created
    const layoutLoadPromiseReject = ref((reason: any) => {
        console.warn(reason);
    });
    const layoutLoadPromise = ref<Promise<void> | null>(null);

    // global ui
    const repairButtonFocus = (e: Event) => {
        //修复element-plus的button坑，垃圾
        let target = e.target as any;
        if (target.nodeName == "SPAN") {
            target = target.parentNode as any;
        }
        target.blur();
    };

    // route related
    const urlList = ref<Array<UrlData>>([]);
    const routeLoading = ref<LoadingInstance>();
    const setRouteLoading = () => {
        routeLoading.value = ElLoading.service({
            lock: true,
            body: true,
            fullscreen: true,
            text: "Loading Route",
            spinner: "el-icon-loading",
            // background: "rgba(0, 0, 0, 0.7)",
            background: "white",
            customClass: "route-loader",
        });
    };
    const clearRouteLoading = () => {
        routeLoading.value!.close();
        routeLoading.value = undefined;
    };

    return {
        defaultRankEmbDiffAlgo,
        defaultPolarEmbDiffAlgo,
        defaultPolarTopoDistAlgo,
        defaultCompViewNames,
        defaultSingleViewNames,

        defaultNodesSelectionEntryMapper,
        getViewSourceNodesSelectionEntry,
        getViewTargetNodesSelectionEntry,
        defaultNodesSelectionEntryDescriptions,

        symbolHops,
        symbolPathByHopAndParentIndex,
        symbolNames,

        defaultHops,
        datasetList,
        fetchOriginDataset,
        getDatasetByName,
        addDataset,
        removeDataset,

        singleDashboardList,
        getSingleDashboardById,
        recentSingleDashboardList,
        // representedSingleDashboardList,
        renderableRecentSingleDashboards,
        recentLen,
        maxRecentLen,
        minRecentLen,
        toSingleDashboardById,
        addSingleDashboard,

        compDashboardList,
        recentCompDashboardList,
        renderableRecentCompDashboards,
        getCompDashboardById,
        toCompDashboardById,
        addCompDashboard,

        initialNewView,
        getTypeReducedDashboardById,
        getTypedDashboardById,
        getViewIndexByName,
        getViewByName,
        insertViewAfterName,

        getPrincipalViewOfDashboard,
        removeSnapshotsOfDashboard,
        calcSnapshotsOfDashboard,
        calcPrincipalSnapshotOfDashboard,
        getPrincipalSnapshotOfDashboard,

        calcAggregatedViewData,

        repairButtonFocus,
        dashboardsLayoutMode,
        globalClearSelMode,
        globalWhenToResizeMode,
        settingsMenuTriggerMode,
        globalLabelType,

        layoutLoadPromiseReject,
        layoutLoadPromise,

        urlList,
        routeLoading,
        setRouteLoading,
        clearRouteLoading,

        testWatcher,
    };
});
