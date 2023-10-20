import {
    taskTypes,
    labelTypes,
    rankEmbDiffAlgos,
    polarEmbDiffAlgos,
    polarTopoDistAlgos,
    dashboardsLayoutModes,
    clearSelModes,
    whenToRescaleModes,
    settingsMenuTriggerModes,
    symbolNames,
} from "@/stores/enums";
import type { DefineComponent } from "vue";
import type * as d3 from "d3";
import type Hexbin from "d3-hexbin";
import type { calcPolar, calcRank3 } from "@/utils/graphUtils";
import type {
    UseWebWorkerFnReturn,
    WebWorkerStatus,
    useWebWorkerFn,
} from "@/utils/myWebWorker";

export type Type_DashboardType = "single" | "comparative";
export type Type_ViewName = string;

export type Type_LabelTypes = (typeof labelTypes)[number];
export type Type_TaskTypes = (typeof taskTypes)[number];

export type Type_RankEmbDiffAlgos = (typeof rankEmbDiffAlgos)[number];
export type Type_polarEmbDiffAlgos = (typeof polarEmbDiffAlgos)[number];
export type Type_polarTopoDistAlgos = (typeof polarTopoDistAlgos)[number];

export type Type_DashboardsLayoutMode = (typeof dashboardsLayoutModes)[number];
export type Type_ClearSelMode = (typeof clearSelModes)[number];
export type Type_WhenToRescaleMode = (typeof whenToRescaleModes)[number];
export type Type_SettingsMenuTriggerMode =
    (typeof settingsMenuTriggerModes)[number];

export type Type_SymbolName = (typeof symbolNames)[number];

/**
 * 选出的nodes的不同集合、条目的id\
 * ```js
 * {\
 *  entryID1: { nodeID1:true, nodeID3:true, ...}\
 *  entryID2: { nodeID5:true, nodeID77:true, ...}\
 * }\
 * ```
 * 这里的`entryID1`, `entryID2`即此类型
 */
export type Type_NodesSelectionEntryId = string;

export interface Dataset {
    /**
     * 数据集（即一个model result）名称。\
     * 也用作路由地址的一部分
     */
    name: string;

    /**
     * 任务类型
     */
    taskType: Type_TaskTypes;
    /**
     * 数据是否完整
     */
    isComplete: boolean;

    /**
     * 预测标签\
     * 根据taskType不同，可能为空。link-prediction，graph-classification一般没有。\
     */
    predLabels: Array<number>;
    /**
     * 真实标签\
     * 根据taskType不同，link-prediction，graph-classification可能没有。\
     * 如果是comparative，为了节省空间，其中一个dataset可能没有
     */
    trueLabels?: Array<number>;

    /**
     * 结点的分类数量
     */
    numNodeClasses: number;

    /**
     * node的颜色映射\
     * 如果是comparative，为了节省空间，其中一个Dataset可能没有
     */
    colorScale?: d3.ScaleOrdinal<string, string>;

    /**
     * 图的数量\
     * 如果是comparative，为了节省空间，其中一个Dataset可能没有
     */
    numGraphs?: number;

    /**
     * hop数\
     * 如果是comparative，为了节省空间，其中一个Dataset可能没有
     */
    hops?: number;
    /**
     * 节点列表\
     * 如果是comparative，为了节省空间，其中一个Dataset可能没有
     */
    nodes?: Array<Node>; //
    /**
     * 最初的结点和图id映射。即某id的node所在的graph的id
     */
    globalNodesDict: Record<
        Type_NodeId,
        // | Type_GraphId
        // | boolean
        {
            gid: Type_GraphId; //the graph that the node is affiliated to
            parentDbIndex: number; //usually 0 or 1;
            hop: number; //0: step1Neighbor, 1: step2Neighbor, -1:originSelf
        }
    >;
    /**
     * 边列表\
     * 如果是comparative，为了节省空间，其中一个Dataset可能没有
     */
    links?: Array<Link>;
    /**
     * 最初的边id和图id映射。即某id的边所在graph的id
     */
    globalLinksDict: Record<Type_LinkId, Type_GraphId>;
    /**
     * 最初的边id和图富信息的映射。即某id的边和其source、target等信息，方便O(1)查找
     */
    globalLinksRichDict: Record<Type_LinkId, Link>;
    /**
     * 每个节点，有哪些边通过哪些点于这个节点直接相连\
     * 如果是comparative，为了节省空间，其中一个Dataset可能没有
     */
    nodeMapLink?: Array<Array<NodeMapLinkEntry>>;

    /**
     * 对每个节点，所有hop的邻居组成的mask\
     * 使用BitSet生成，可参考 https://github.com/infusion/BitSet.js\
     * 如果是comparative，为了节省空间，其中一个Dataset可能没有
     */
    neighborMasks?: Array<string>; //
    /**
     * 对每个hop，对于每个节点的邻居组成的mask\
     * 注意这是个累积hop，即1hop的，12hop的，123hop的，...\
     * 使用BitSet生成，可参考 https://github.com/infusion/BitSet.js\
     * 如果是comparative，为了节省空间，其中一个Dataset可能没有
     */
    neighborMasksByHop?: Array<Array<string>>;
    /**
     * 对每个hop，对于每个节点的邻居组成的mask\
     * 注意仅包含当前hop
     * 使用BitSet生成，可参考 https://github.com/infusion/BitSet.js\
     * 如果是comparative，为了节省空间，其中一个Dataset可能没有
     */
    neighborMasksByHopPure?: Array<Array<string>>;
    /**
     * 使用force-directed layout生成的节点和边的坐标信息\
     * 如果没有提前算好的，可能没有\
     * 如果是comparative，为了节省空间，其中一个Dataset可能没有
     */
    graphCoordsRet?: Array<d3.SimulationNodeDatum & NodeCoord>;

    /**
     * 结点特征的维度。1则为dense features
     */
    numNodeSparseFeatureDims: number;
    /**
     * 结点特征。数组长度等于结点长度\
     * 注意dense feature: 可能每个subArr的长度为1\
     * sparse feature: 只记录非空，每个subArr每个元素对应原结点的索引由nodeFeatureIndexes给出
     */
    nodeSparseFeatureValues: Array<Array<number>>;
    /**
     * 仅sparse features\
     * 对于每个结点，结点特征只记录非空\
     * 故此数组的每个subArr，对应了这个结点的非空特征在特征维度中的索引
     */
    nodeSparseFeatureIndexes: Array<Array<number>>;
    /**
     * 长度为结点特征的维度数\
     * 记录了每个维度的实际物理含义是什么\
     * 某些数据集可能没有
     */
    nodeSparseFeatureIndexToSemantics: Record<number, string>;
    /**
     * dense feature
     */
    denseNodeFeatures: d3.DSVParsedArray<{
        [key: string]: unknown;
        id: Type_NodeId;
    }>;
    // Array<{ id: Type_NodeId; feature: number }>;

    /**
     * node embeddings
     */
    embNode: Array<Array<number>>;
    /**
     * node embeddings使用tsne降维的结果，如果没有提前算好的，可能没有
     */
    tsneRet?: Array<TsneCoord>;

    /**
     * 仅连接预测。含义是原图中存在，且预测正确的边，\
     * 用一个点对表示，即[边的出发点id, 边的终点id]
     */
    trueAllowEdges: [Type_NodeId, Type_NodeId][];
    /**
     * 仅连接预测。含义是原图中存在，且预测错误的边，\
     * 用一个点对表示，即[边的出发点id, 边的终点id]。\
     * 注意这是原图中的边(ground truth)，而不是预测的结果
     */
    falseAllowEdges: [Type_NodeId, Type_NodeId][];
    /**
     * 仅连接预测。原图中没有的但推荐出来的边由一个分数排序决定，这个数字表示取前几名

     */
    trueUnseenTopK: number;
    /**
     * 仅连接预测。原图中没有的但推荐出来的边，
     * 每字段一个表示一个结点id，其值为Array，长度为"trueUnseenTopK"，\
     * 表示给这个节点推荐的边，按分数由高到低排序
     */
    trueUnseenEdgesSorted: Record<Type_NodeId, Type_NodeId[]>;

    /**
     * 仅图分类。图的类别个数
     */
    numGraphClasses: number;
    /**
     * 仅图分类。每个图有哪些结点和边。使用Record形式
     */
    graphRecords: Record<
        Type_GraphId,
        {
            nodesRecord: Record<Type_NodeId, Type_GraphId>;
            linksRecord: Record<Type_LinkId, Type_GraphId>;
        }
    >;
    /**
     * 仅图分类。每个图有哪些结点和边，使用数组形式。
     */
    graphArr: Array<{
        gid: Type_GraphId;
        nodes: Array<Type_NodeId>;
        links: Array<Type_LinkId>;
    }>;

    /**
     * 仅图分类。graph的 trueLabels
     */
    graphTrueLabels: number[];
    /**
     * 仅图分类。graph的 predLabels
     */
    graphPredLabels: number[];
    /**
     * 仅图分类。图的类别颜色映射
     */
    graphColorScale: d3.ScaleOrdinal<string, string>;
    /**
     * 仅图分类。图的索引，这是为了和下面的两个labels对应。\
     * 因为可能有shuffle的情况
     */
    graphIndex: Type_GraphId[];
    /**
     * 仅图分类。用于描述phase中各数字对应什么
     */
    phaseDict: Record<number, string>;
    /**
     * 仅图分类。每个图的训练阶段（train，validate, test)\
     * 为节省空间用数字表示，顺序和graphIndex对应
     */
    phase: number[];
    /**
     * 仅图分类。graph的embeddings
     */
    embGraph?: number[][];
    /**
     * 仅图分类。graph的embeddings通过tsne算法降至2维的结果
     */
    graphTsneRet?: Array<TsneCoord>;
    /**
     * 仅图分类。异步加载的图的特征。
     */
    graphUserDefinedFeatureRecord: Record<
        Type_GraphFeatureName,
        Record<Type_GraphId, number>
    >;
}

/**
 * 一个view的各种定义和数据。\
 * header和body的都有，甚至包括了header和组件和body的组件是什么。\
 * by doing so, we treat component as data to some degree.\
 * isShowLinks?: boolean; 这个算props还是算view的属性？\
 * 思考问题：只有有必要提取到公共时才提取（例如至少两个不同view用到同属性），否则用各自的props ?\
 * 注意！我们约定，并不是看公共不公共，\
 * 而是看这个属性对于确定唯一的view，是否必要（类似于id、key、主键的性质），\
 * 非必要不用props，而是用ts+store扩展，\
 * 例如tsneView的属性`{which?: number;}`在db运行时确定，用于区分两个可能有相同的名称的view，故它是带有id性质的\
 */
export interface View {
    /**
     * view名称。不一定是唯一的
     */
    viewName: Type_ViewName;

    /**
     * header的组件对象
     */
    headerComp: ShallowRef<InstanceType<typeof DefineComponent>>;
    /**
     * body的组件对象，其`<template>`顶层一般是svg
     */
    bodyComp: ShallowRef<any>;
    /**
     * 不同view的body可能需要的独特数据\
     * 但容易使代码混乱，很少用
     */
    bodyProps: any;
    /**
     * 不同view的header可能需要的独特数据\
     * 例如早期版本，tsne中是否showLink\
     * 但容易使代码混乱，现在很少用
     */
    headerProps: any; //

    /**
     * view的总宽度，包括header+body
     * 创建db时，会initialNewView，会设置这个值，有默认的，有独特的\
     * 在db中经过一些异步计算（Promise），可能会修改view的这个值\
     */
    initialWidth: number; //NOTE
    /**
     * view的总高度，包括header+body
     * 创建db时，会initialNewView，会设置这个值，有默认的，有独特的\
     * 在db中经过一些异步计算（Promise），可能会修改view的这个值\
     */
    initialHeight: number; //NOTE db中，创建view或者修改view也可以设置这个值

    /**
     * 仅body的width，通常用于svg\
     * 此尺寸替代了svg组件中的props，被用于大量尺寸、坐标相关的计算
     */
    bodyWidth: number;
    /**
     * 仅body的height，通常用于svg\
     * 此尺寸替代了svg组件中的props，被用于大量尺寸、坐标相关的计算
     */
    bodyHeight: number;
    /**
     * body组件的margin，通常用于svg\
     * 注意！尽量使用百分比margin，以防止负数
     */
    bodyMargins: {
        top: number;
        right: number;
        bottom: number;
        left: number;
    };

    /**
     * 在view组件卸载时应当进行的调用。\
     * 可能是一些手动的垃圾回收\
     * 注意，因为子组件先于父组件unmount，所以这个放在onBeforeUnmount
     */
    onBeforeUnmountCallbacks?: Array<() => void>;

    /**
     * resizeEnd时的信号，用boolean进行是非的翻转即可
     */
    resizeEndSignal: boolean;
    /**
     * 初始时brush是否开启。否则开启zoom拖拽。\
     * zoom滚轮放缩一直有效
     */
    isBrushEnabled: boolean;
    /**
     * 启用brush的函数ref
     */
    brushEnableFunc: Ref<() => void> | (() => void);
    /**
     * brush的函数ref
     */
    brushDisableFunc: Ref<() => void> | (() => void);
    /**
     * 启用zoom拖拽的函数ref\
     * zoom滚轮放缩一直有效
     */
    panEnableFunc: Ref<() => void> | (() => void);
    /**
     * 禁用zoom拖拽的函数ref\
     * zoom滚轮放缩一直有效
     */
    panDisableFunc: Ref<() => void> | (() => void);
    /**
     * 隐藏brush出来的rect的函数ref
     */
    hideRectWhenClearSelFunc: Ref<() => void> | (() => void);
    /**
     * 恢复zoom原始视野的函数ref
     */
    resetZoomFunc: Ref<() => void> | (() => void);

    /**
     * dashboard是否正在获取某view的snapshot\
     * 放在这里只是为了数据结构上便于管理。
     */
    isGettingSnapshot: boolean;
    gettingSnapShotError: unknown;
    snapshotBase64: string;

    /**
     * 用于设置view对象的attr和value，并且返回本身，便于链式调用
     * @param attr
     * @param value
     */
    setAttr: <K extends keyof View>(attr: K, value: any) => View;
}

/**
 * View with coords, namely, `x`, `y`, and `id`\
 * 在view中我们不命名为“points”，而是”coords“。\
 * points是个模糊概念。view中是坐标（用来渲染的），db中是selection，是一些Record(即dict)，用来处理集合上的和选择上的逻辑的\
 * 如何从db的初始selection得到坐标，是一个接口逻辑。
 * 针对renderCoords进行brush，如何影响selection，亦成为外部的接口即可，而不必写进brush的功能里
 */
export interface NodeView<T extends NodeCoord> extends View {
    isShowHopSymbols: boolean;
    nodeRadius: number;
    sourceCoords: T[];
    rescaledCoords: T[];
    isRelativeColorScale: boolean;
}

export interface LinkableView extends View {
    isShowLinks: boolean;
    linkOpacity: number;
}

export interface AggregatedView extends NodeView<T> {
    isShowAggregation: boolean;
    isShowMesh: boolean;

    meshRadius: number;
    hexbin: Hexbin<any & NodeCoord>;
    clusterRadiusScale: d3.ScaleRadial;
    /**
     * 用来查找某node在哪个cluster
     */
    pointCluster: Record<Type_NodeId, Type_ClusterId>;
    clusters: {
        id: string;
        pointIds: Type_NodeId[];
        pointCenter: {
            x: number;
            y: number;
        };
        hexCenter: {
            x: number;
            y: number;
        };
        count: Record<string | number | symbol, number>;
    }[];

    //NOTE 为什么aggregatedCoords和clusters是分开的,仅仅为了可读性吗
    //   coords命名，带有明显的几何风格，而clusters则带有一些集合风格，一些数据风格

    aggregatedCoords: {
        id: Type_NodeId;
        x: number;
        y: number;
        mag: number; //magnitude
        arcs: Array<d3.PieArcDatum<any>>;
    }[];
    aggregatedLinks: AggregatedLink[];
}

/**
 * PolarView定义\
 * REVIEW 此interface相比于之前的更加具体(之前的更抽象，不带有具体名字)\
 */
export interface PolarView extends AggregatedView, LinkableView {
    polarData: Array<PolarDatum>; //[{id,hop,embDiff,topoDist}, {id,hop,embDiff,topoDist}, ...]
    polarEmbDiffAlgo: "single" | "average" | "center";
    polarTopoDistAlgo: "shortest path" | "hamming" | "jaccard";
    hops: number;

    /**
     * 最大圆半径
     */
    R: number;

    /**
     * 由hop获取radius的范围，跨组件要用
     */
    // getRadiusRangeByHop: (hop: number) => [number, number];

    /**
     * @override
     * 由polarData经过初步map得到的初始坐标，即字段变换得到的初始坐标
     */
    sourceCoords: PolarCoord[];

    /**
     * @deprecated
     */
    rescaledCoords: never;

    /**
     * 与那些直接是笛卡尔坐标系的view不同\
     * 这里每次rescale必须用极坐标\
     * isShowAggregation切换至false或resize，都rescale\
     */
    rescaledPolarCoords: PolarCoord[];

    /**
     * 从rescaledPolarCoords得到\
     * 非aggregate模式，用这个渲染
     */
    cartesianCoords: (PolarCoord & NodeCoord)[];

    /**
     * 此view中用到的link
     */
    localLinks: Link[];

    linkRecord: Record<Type_LinkId, Link>;

    isUsingSecondDbPredLabel: boolean;
    /**
     * @extends
     */
    // aggregatedCoords
    //每次切换到aggregate模式，重新计算这个，用这个渲染
}

export interface RankDatum {
    id: Type_NodeId;
    r1: number;
    r2: number;
    d1: number;
    d2: number;
}
/**
 * RankView定义\
 * REVIEW 此interface相比于之前的更加具体(之前的更抽象，不带有具体名字)\
 */
export interface RankView extends AggregatedView {
    rankEmbDiffAlgo: "single" | "average" | "center";
    rankData: Array<RankDatum>;
    isUsingSecondDbPredLabel: boolean;
}
export interface DenseView extends View {
    isRelative: boolean;
    isCorrespondAlign: boolean;
    numColumns: number;
    subHeight: number;
}

/**
 * 有些view需要在body和head同时拥有处理loading、error、result的能力，故状态提升
 */
export interface AsyncView extends View {
    isLoading: boolean;
    loadingProcess: (any) => Promise<any>;
    loadingError: any;
}
export interface MultiGraphView extends LinkableView, NodeView<T> {
    subHeight: number;
    isAlignHeightAndWidth: boolean;
    numColumns: number;

    /**
     * 用数组存放了graph+nodesId+nodesCoords+linkId信息。\
     * 每个graph为一个group，相对于左上角,且经过了subWidth，subHeight的scale
     */
    groupedRescaledCoords: Array<{
        gid: Type_GraphId;
        filteredLinks: Type_LinkId[];
        filteredNodes: NodeCoord[];
    }>;
}
export interface SparseView extends View {
    diffColorRange: [number, number];
    selColorRange: [number, number];
    sel0ColorRange: [number, number]; //only in dbWiseComp
    sel1ColorRange: [number, number]; //only in dbWiseComp
    isAdaptiveColorInterpolate: boolean;
}
export interface LinkPredView extends LinkableView {
    isShowNeighbors: boolean;
    currentHops: number;

    symbolUnseen: d3.SymbolType;
    symbolSelection: d3.SymbolType;

    // symbolHops: d3.SymbolType[];//用myStore全局的

    nodeRadius: number;
    isShowGroundTruth: boolean;
    isShowTrueAllow: boolean;
    isShowFalseAllow: boolean;
    isShowTrueUnseen: boolean;
    numTrueUnseen: number;

    isShowSelfLoop: boolean;
}
export interface Dashboard {
    id: string;
    name: string;
    date: string | number | Date;

    isRepresented: boolean;

    /**
     * 是否为根db，即最初的db
     */
    isRoot: boolean;

    /**
     * 此db的初始数据是否完整
     */
    isComplete: boolean;

    /**
     * 生成此db的父亲db的id
     */
    parentId: string | [string, string];

    /**
     * 从上一个的db的哪个view选出来的。对于生成view的略缩图用来表征整个Dashboard非常重要
     */
    fromViewName: Type_ViewName;

    /**
     * 图的初始结点坐标，通常是force-directed
     */
    graphCoordsRet:
        | Array<Node & d3.SimulationNodeDatum & NodeCoord>
        | Array<d3.SimulationNodeDatum & NodeCoord>;

    /**
     * 此dashboard所含节点，（从上一个db或者从dataset得到）
     */
    srcNodesDict: Record<
        Type_NodeId,
        // | Type_GraphId
        // | boolean
        {
            gid: Type_GraphId; //the graph that the node is affiliated to
            parentDbIndex: number; //usually 0 or 1;
            hop: number; //0: step1Neighbor, 1: step2Neighbor, -1:originSelf
        }
    >;

    /**
     * 此dashboard所含 linkIds ，（从上一个db或者从dataset得到）
     */
    srcLinksDict: Record<Type_LinkId, Type_GraphId>; //NOTE we don't use undefined, use '' instead
    srcLinksArr: Link[];

    /**
     * 用于保存views定义的列表
     */
    viewsDefinitionList: Array<View>;

    /**
     * 手动清除还是自动清除，即每次合并选择还是重新选择，仅当前db
     */
    clearSelMode: Type_ClearSelMode;
    /**
     * label类型
     */
    labelType: Type_LabelTypes;

    /**
     * 当此db的view进行resize的时候，何时重新计算svg中的坐标，瞬时或resize结束时\
     * resize结束时利用一个小的延时器实现
     */
    whenToRescaleMode: Type_WhenToRescaleMode;

    /**
     * 点击restoreViewsSizes按钮,此db所有的view的大小恢复成默认大小\
     * 使用这样一个信号实现.每次点击按钮反转true和false即可
     */
    restoreViewsSizesSignal: boolean;

    /**
     * db刚创建时用于计算的Promise
     */
    calcOnCreatedPromise: Promise<any>;

    /**
     * 选出的nodes的不同集合、条目\
     * {\
     *  entryID1: { nodeID1:true, nodeID3:true, ...}\
     *  entryID2: { nodeID5:true, nodeID80:true, ...}\
     * }\
     * NOTE 实际运行时，某个entry可能只有一个单点，而不是字典形式\
     * NOTE 编程式添加一个view，关于nodesSelection，需要改：mapper，description，以及db里的这个nodesSelection
     */
    nodesSelections: Record<
        Type_NodesSelectionEntryId,
        Record<
            Type_NodeId,
            // | Type_GraphId
            // | boolean
            {
                gid: Type_GraphId; //the graph that the node is affiliated to
                parentDbIndex: number; //usually 0 or 1;
                hop: number; //0: step1Neighbor, 1: step2Neighbor, -1:originSelf
            }
        >
        //REVIEW
        //         when boolean: one graph, one dashboard
        //         when Type_GraphId: multi graph, esp. graph-classification
        //         when {...}   :  dashboard-wise comparison
        //{gid:Type_GraphId, parentDbIndex:number, hop: -1}
    >;

    /**
     * hover一个view中的一个或多个node(s)或graph(s)时，其他view是否也相应地高亮显示此node(s)。
     */
    isHighlightCorrespondingNode: boolean;
    /**
     * 高亮的nodeID
     */
    highlightedNodeIds: Record<Type_NodeId, Type_GraphId | {}>;
    /**
     * 高亮的graphID, 贪婪模式。只要高亮了单个node，就将它的graph计算在内
     */
    highlightedGreedyGraphIds: Record<Type_GraphId, boolean | {}>;
    /**
     * 高亮的graphID, 完整模式。一个完整的graph（全部结点）高亮
     */
    highlightedWholeGraphIds: Record<Type_GraphId, boolean | {}>;
}

export interface DbWiseComparativeDb extends Dashboard {
    /**
     * @override
     */
    parentId: [string, string];
}

export interface SingleDashboard extends Dashboard {
    /**
     * 参照的dataset的名字，即原始数据
     */
    refDatasetName: string;

    tsneCalcPromise: Promise<void>;
    tsneCalcPromiseReject: (reason: any) => void;
    /**
     * tsne的降维结果 \
     * NOTE maybe shallow copy?
     */
    tsneRet: Array<TsneCoord>;

    graphTsneCalcPromise: Promise<void>;
    graphTsneCalcPromiseReject: (reason: any) => void;
    /**
     * graphTsne的降维结果 \
     * NOTE maybe shallow copy?
     */
    graphTsneRet: Array<GraphTsneCoord>;
}

export interface CompDashboard extends Dashboard {
    tsne1CalcPromise: Promise<void>;
    tsne1CalcPromiseReject: (reason: any) => void;
    tsne2CalcPromise: Promise<void>;
    tsne2CalcPromiseReject: (reason: any) => void;
    /**
     * 第一个model result的tsne的降维结果\
     * NOTE maybe shallow copy?
     */
    tsneRet1: Array<TsneCoord>;
    /**
     * 第2个model result的tsne的降维结果\
     * NOTE maybe shallow copy?
     */
    tsneRet2: Array<TsneCoord>;

    graphTsne1CalcPromise: Promise<void>;
    graphTsne1CalcPromiseReject: (reason: any) => void;
    graphTsne2CalcPromise: Promise<void>;
    graphTsne2CalcPromiseReject: (reason: any) => void;
    /**
     * 第一个model result的graph tsne的降维结果\
     * NOTE maybe shallow copy?
     */
    graphTsneRet1: Array<GraphTsneCoord>;
    /**
     * 第2个model result的graph tsne的降维结果\
     * NOTE maybe shallow copy?
     */
    graphTsneRet2: Array<GraphTsneCoord>;

    /**
     * 所依赖的dataset们的name
     */
    refDatasetsNames: [string, string];

    /**
     * 用于计算rankView的data的一个worker定义，包含了\
     * workerFn, workerStatus, workerTerminate,\
     * see https://vueuse.org/core/useWebWorkerFn/
     */
    rankWorker: {
        workerFn: (
            ...fnArgs: Parameters<typeof calcRank3>
        ) => Promise<ReturnType<typeof calcRank3>>;
        workerStatus: Ref<WebWorkerStatus>;
        workerTerminate: (status?: WebWorkerStatus) => void;
    }; //NOTE 这里类型要全部拆开重写
    // rankWorker: ReturnType<typeof useWebWorkerFn>;

    /**
     * 用于计算polarView的data的一个worker定义，包含了\
     * workerFn, workerStatus, workerTerminate,\
     * see https://vueuse.org/core/useWebWorkerFn/
     */
    polarWorker: {
        workerFn: (
            ...fnArgs: Parameters<typeof calcPolar>
        ) => Promise<ReturnType<typeof calcPolar>>;
        workerStatus: Ref<WebWorkerStatus>;
        workerTerminate: (status?: WebWorkerStatus) => void;
    };
    // polarWorker: ReturnType<typeof useWebWorkerFn<typeof calcPolar>>;
}

export interface RecentDb {
    id: string; //DbId
    renderIndex: number;
    // isRepresented: boolean;
}

/**
 * the data of home layout, each is a model result, aka a dataset in CorGIE-2
 */
export interface UrlData {
    /**
     * the name will be used as route url, so it's identical\
     * usually datasetName-model-otherInfo
     */
    name: string;

    task: Type_TaskTypes;
    date: Date;
    /**
     * dataset name without other info\
     * "dataset" here usually means benchmark dataset in deep learning field\
     * e.g., Cora, Pubmed, Citeseer
     */
    graph: string;
    model: string;
}

export type Type_NodeId = string;
export type Type_LinkId = string;
export type Type_GraphId = string;

export interface Node {
    id: Type_NodeId;
    label: number;
    /**
     * only in multi graphs
     */
    gid: Type_GraphId;
}

export interface NodeCoord {
    id: Type_NodeId;
    x: number;
    y: number;
}

export interface TsneCoord extends NodeCoord {}

export interface GraphNodeCoord {
    id: Type_GraphId;
    x: number;
    y: number;
}

export interface GraphTsneCoord extends GraphNodeCoord {}

/**
 * d3-force formatted node coord type
 */
// export type NodeGraphCoord = d3.SimulationNodeDatum & {
//     id: Type_NodeId;
//     index: number;
//     x: number;
//     y: number;
//     vx: number;
//     vy: number;
// };

export interface Link {
    eid: Type_LinkId;
    /**
     * only in multi graphs
     */
    gid: Type_GraphId;
    /**
     * some dataset may contain link labels
     */
    label?: number;
    source: Type_NodeId;
    target: Type_NodeId;
}
/**
 * d3-force formatted link coord type
 */
// export type LinkGraphCoord = d3.SimulationLinkDatum<NodeGraphCoord & Node> & {
//     index: number;
//     source: NodeGraphCoord & Node;
//     target: NodeGraphCoord & Node;
// };

/**
 * 由fetch得到的原始数据格式
 */
export interface FetchedGraphData {
    directed: boolean;
    multigraph: boolean;
    graphs:
        | Array<{
              id: Type_GraphId;
              label: number;
              nodes: Array<Type_NodeId>;
              edges: Array<Type_LinkId>;
          }>
        | Record<
              number,
              {
                  id: Type_GraphId;
                  label: number;
                  nodes: Array<Type_NodeId>;
                  edges: Array<Type_LinkId>;
              }
          >;
    nodes: Array<Node>;
    edges: Array<Link>;
    hops?: number;
}

/**
 * nodeMapLink(nodeMapLink)每一条的类型，即哪些个点、边和当前索引的点相连
 */
export interface NodeMapLinkEntry {
    nid: Type_NodeId;
    eid: Type_LinkId;
}

/**
 * @deprecated
 * wrap all coords calculated by d3-force\
 * may be loaded from file as well
 */
export type GraphCoords = {
    nodes: Array<Node & d3.SimulationNodeDatum & NodeCoord>;
    links: Array<
        Link & d3.SimulationLinkDatum<d3.SimulationNodeDatum & Node & NodeCoord>
    >;
};

export type Type_ClusterId = string;
export type Type_AggregatedLinkId = string;
export interface AggregatedLink {
    aeid: Type_AggregatedLinkId;
    source: Type_ClusterIndex;
    target: Type_ClusterIndex;
    baseLinks: Array<Link>;
}

export interface PredDataCommon {
    taskType: Type_TaskTypes;
}
export interface NodeClassificationPredData extends PredDataCommon {
    // isNodeClassification: true;//废弃
    numNodeClasses: number;
    predLabels: number[];
    trueLabels: number[];
}
export interface LinkPredictionPredData extends PredDataCommon {
    // isLinkPrediction: true;//废弃
    /**
     * in link prediction tasks, node class is not necessary
     */
    numNodeClasses?: number;
    trueAllowEdges: [Type_NodeId, Type_NodeId][];
    falseAllowEdges: [Type_NodeId, Type_NodeId][];
    trueUnseenTopK: number;
    trueUnseenEdgesSorted: Record<Type_NodeId, Type_NodeId[]>;
}
export interface GraphClassificationPredData extends PredDataCommon {
    // isGraphClassification: true;//废弃
    /**
     * in graph classification tasks, node class is not necessary
     */
    numNodeClasses?: number;
    numGraphClasses: number;
    graphIndex: Type_GraphId[];
    predLabels: number[];
    trueLabels: number[];
    phaseDict?: Record<number, string>;
    phase: number[] | string[]; //可能直接就是string了，需要我们手动处理
    graphEmbeddings: number[][];

    [key: string]: unknown;
}

export interface PolarDatum {
    id: Type_NodeId;
    embDiff: number;
    hop: number;
    topoDist: number;
}
export interface PolarCoord {
    id: Type_NodeId;
    angle: number;
    radius: number;
    hop?: number;
}
