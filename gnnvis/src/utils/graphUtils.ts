import * as d3 from "d3";
import { hexbin, type Hexbin } from "d3-hexbin";
import BitSet from "bitset";
import type {
    Node,
    Link,
    NodeMapLinkEntry,
    Type_NodeId,
    PolarCoord,
    NodeCoord,
    Type_ClusterId,
    AggregatedLink,
    PolarDatum,
    RankDatum,
    Type_GraphId,
    Type_LinkId,
} from "@/types/types";
////////////////////////////////////////////////////////////////////////////////
//////////// SECTION - worker funcs, write here just to eliminate lint errors
import tsnejs from "tsne";
const sim2dist = d3.scaleLinear().domain([-1, 1]).range([1, 0]);
const getCosineDistance = (u: Array<number>, v: Array<number>) => {
    let p = 0,
        magU = 0,
        magV = 0;
    for (let i = 0; i < u.length; i++) {
        p += u[i] * v[i];
        magU += Math.pow(u[i], 2);
        magV += Math.pow(v[i], 2);
    }
    const mag = Math.sqrt(magU) * Math.sqrt(magV);
    const sim = p / mag;
    return sim2dist(sim);
};
const getJaccardDistance = (mask1: string | BitSet, mask2: string | BitSet) => {
    const bs1 = typeof mask1 === "string" ? new BitSet("0x" + mask1) : mask1;
    const bs2 = typeof mask2 === "string" ? new BitSet("0x" + mask2) : mask2;
    const intersection = bs1.and(bs2).cardinality();
    const union = bs1.or(bs2).cardinality();
    return union === 0 ? 0 : 1 - intersection / union;
};
const getHammingDistance = (mask1: string | BitSet, mask2: string | BitSet) => {
    const bs1 = typeof mask1 === "string" ? new BitSet("0x" + mask1) : mask1;
    const bs2 = typeof mask2 === "string" ? new BitSet("0x" + mask2) : mask2;
    return bs1.xor(bs2).cardinality(); //bitset中arr中元素的个数
};
const calcRank = (distances1: number[], distances2: number[]) => {
    const getRanks = (distances: { id: number; d: number }[]) => {
        const sorted = distances.sort((a, b) => a.d - b.d);
        return sorted
            .map((x, i) => ({ ...x, r: i }))
            .sort((a, b) => a.id - b.id);
    };
    const ranks1 = getRanks(distances1.map((d, i) => ({ id: i, d }))); //the d is named as d
    const ranks2 = getRanks(distances2.map((d, i) => ({ id: i, d })));

    //ANCHOR - definition related
    const ranksMerge = ranks1.map((r1, i) => ({
        id: i + "",
        d1: r1.d,
        r1: r1.r,
        d2: ranks2[i].d,
        r2: ranks2[i].r,
        // pl1: predLabels1[i],
        // pl2: predLabels2[i],
        // tl: trueLabels[i],
    }));
    return ranksMerge;
};

/**
 * compute the shortest path between a node and a node group // BFS
 * @param {*} node
 * @param {*} targetBS
 * @param {*} nodeMapLink
 * @returns
 */
function computeShortestPath(
    node: number,
    targetBS: BitSet,
    nodeMapLink: Array<Array<NodeMapLinkEntry>>
) {
    if (targetBS.get(node) === 1) return -1;
    const q = [node];
    const prev = new Array(nodeMapLink.length).fill(-1);
    const visited = { [node]: true };
    while (q.length > 0) {
        let cur: number = q.shift()!;
        for (const e of nodeMapLink[cur]) {
            if (targetBS.get(+e.nid) === 1) {
                // shortest path found
                const sp = [e.nid, cur];
                while (prev[cur] !== -1) {
                    cur = prev[cur];
                    sp.push(cur);
                }
                return sp;
            }
            if (!visited[+e.nid]) {
                visited[+e.nid] = true;
                prev[+e.nid] = cur;
                q.push(+e.nid);
            }
        }
    }
    return -1;
}
//////////// !SECTION - worker funcs, write here just to eliminate lint errors
////////////////////////////////////////////////////////////////////////////////
export const isEmptyDict = (dict: any) => {
    // return Object.keys(dict).length>0//NOTE O(n)

    //NOTE O(1):
    for (const i in dict) {
        // 如果不为空，则会执行到这一步，返回
        return false;
    }
    return true;
};

/**
 * 某个obj为nodeId对应graphId，转换成graphId（不重不漏）各有哪些nodeId
 * @param nodeMapGraph 一个{}，每个nodeId对应graphId
 */
export function nodeMapGraph2GraphMapNodes(
    nodeMapGraph: Record<Type_NodeId, Type_GraphId>
): Record<Type_GraphId, Record<Type_NodeId, Type_GraphId>>;

export function nodeMapGraph2GraphMapNodes<
    T extends {
        gid: Type_GraphId;
    }
>(
    nodeMapGraph: Record<Type_NodeId, T>
): Record<Type_GraphId, Record<Type_NodeId, T>>;

export function nodeMapGraph2GraphMapNodes<
    T extends {
        gid: Type_GraphId;
    }
>(
    nodeMapGraph: Record<Type_NodeId, T | Type_GraphId>
): Record<Type_GraphId, Record<Type_NodeId, T | Type_GraphId>> {
    {
        const ret: Record<
            Type_GraphId,
            Record<Type_NodeId, Type_GraphId | T>
        > = {};
        for (const nodeId in nodeMapGraph) {
            const graphInfo = nodeMapGraph[nodeId];
            if (typeof graphInfo === "string") {
                ret[graphInfo] = {
                    ...(ret[graphInfo] || {}),
                    [nodeId]: graphInfo,
                };
            } else {
                const { gid: graphId } = graphInfo;
                ret[graphId] = {
                    ...(ret[graphId] || {}),
                    [nodeId]: graphInfo,
                };
            }
        }
        return ret;
    }
}

/**
 * 某obj为一些graphId各对应的nodeId有哪些。\
 * 此函数用于转换成这个obj中所有nodeId（不重不漏）分别在哪个graphId\
 * 并且额外信息予以保留
 * @param graphMapNodes
 */
export const graphMapNodes2nodeMapGraph = <T extends {}>(
    graphMapNodes: Record<Type_GraphId, Record<Type_NodeId, T>>
): Record<Type_NodeId, T & { gid: Type_GraphId }> => {
    const nodeMapGraph: Record<Type_NodeId, T & { gid: Type_GraphId }> = {};
    for (const graphId in graphMapNodes) {
        for (const nodeId in graphMapNodes[graphId])
            nodeMapGraph[nodeId] = {
                ...graphMapNodes[graphId][nodeId],
                gid: graphId,
            };
    }
    return nodeMapGraph;
};

/**
 * Compute the neighborMasksByHop and neighborMasks (all hops combined)
 *    nodeMapLink is an array of array: mapping source id to an array of targets,
 *    where a target is an object with node id and edge id
 * @param  numNodes
 * @param  nodeMapLink
 * @param  hops
 * @returns
 */
export const computeNeighborMasks = (
    numNodes: number,
    nodeMapLink: Array<Array<NodeMapLinkEntry>>,
    hops: number
) => {
    const masks = [],
        masksByHop = [],
        masksByHopPure = [];
    let last: BitSet[];
    for (let i = 0; i < numNodes; i++) {
        masks.push(new BitSet("0x0")); //NOTE 这里其实是空。
        // NOTE include self
        masks[i].set(i, 1);
    }

    // first hop
    for (let sid = 0; sid < nodeMapLink.length; sid++) {
        for (const targetNode of nodeMapLink[sid]) {
            const tid = targetNode.nid;
            masks[+sid].set(+tid, 1);
            masks[+tid].set(sid, 1);
        }
    }
    masksByHop.push(masks.map((m) => m.clone())); //NOTE
    masksByHopPure.push(masks.map((m) => m.clone())); //NOTE
    last = masksByHop[0];

    // hop > 1
    for (let h = 1; h < hops; h++) {
        const cur = [];
        for (let i = 0; i < numNodes; i++) {
            cur.push(new BitSet(0));
        }
        for (let i = 0; i < numNodes; i++) {
            const m = masks[i]; //m每次会累积
            for (const sid of m.toArray()) {
                for (const targetNode of nodeMapLink[sid]) {
                    m.set(+targetNode.nid, 1);
                }
            }

            for (const sid of last[i].toArray()) {
                for (const targetNode of nodeMapLink[sid]) {
                    cur[i].set(+targetNode.nid, 1);
                }
            }
        }
        masksByHop.push(cur);
        masksByHopPure.push(cur.map((d, i) => last[i].xor(d).set(i, 0))); //NOTE exclude self
        last = cur;
    }
    const strMasks = masks.map((d) => d.toString(16));
    const strMasksByHop = masksByHop.map((arr) =>
        arr.map((d) => d.toString(16))
    );
    const strMasksByHopPure = masksByHopPure.map((arr) =>
        arr.map((d) => d.toString(16))
    );
    return {
        // neighborMasks: masks,
        neighborMasks: strMasks,
        // neighborMasksByHop: masksByHop
        neighborMasksByHop: strMasksByHop,
        neighborMasksByHopPure: strMasksByHopPure,
    };
};

export const calcNeighborDict = <
    T extends { hop: number } = { hop: number },
    U extends { id: Type_NodeId } = { id: Type_NodeId }
>(
    srcNodesDictOrArr: Record<Type_NodeId, T> | Array<Type_NodeId> | Array<U>,
    hops: number,
    neighborMasksByHop: Array<Array<string>>
) => {
    let selArr: Type_NodeId[];
    if (Array.isArray(srcNodesDictOrArr)) {
        if (srcNodesDictOrArr.length > 0)
            selArr =
                typeof srcNodesDictOrArr[0] === "string"
                    ? (srcNodesDictOrArr as Type_NodeId[])
                    : srcNodesDictOrArr.map((d) => (d as U).id);
        else return {};
    } else {
        if (!isEmptyDict(srcNodesDictOrArr)) {
            selArr = Object.keys(srcNodesDictOrArr).filter(
                (d) => srcNodesDictOrArr[d]
            );
        } else return {};
    }

    const ret: Record<
        Type_NodeId,
        {
            hop: number;
        }
    > = selArr.reduce(
        (acc, cur) => ({
            ...acc,
            [cur]: {
                hop: -1,
            },
        }),
        {}
    );
    for (let hop = 0; hop < hops; ++hop) {
        let curNeighborBS = new BitSet();
        selArr.forEach((idStr) => {
            curNeighborBS = curNeighborBS.or(
                new BitSet("0x" + neighborMasksByHop[hop][+idStr] || undefined)
            );
        });

        if (hop === 0) {
            selArr.forEach((d, i) => curNeighborBS.set(+d, 0)); //排除本身
        }
        if (hop > 0) {
            let lastNeighborBS = new BitSet();
            selArr.forEach((idStr) => {
                lastNeighborBS = lastNeighborBS.or(
                    new BitSet(
                        "0x" + neighborMasksByHop[hop - 1][+idStr] || undefined
                    )
                );
                curNeighborBS = curNeighborBS.xor(lastNeighborBS); //1,2,3...hop阶和1,2,3,...hop-1阶抑或，仅剩hop
            });
        }
        curNeighborBS.toArray().forEach((idNumber) => {
            if (!ret[idNumber + ""]) {
                ret[idNumber + ""] = {
                    hop: hop,
                };
            }
        });
    }
    return ret;
};

/**
 * Filter edges: self loop and duplicates are removed.
 *      Note: we treat all edges as undirectional.
 *      Compute an edge dictionary by its source ID
 *
 * @param  numNodes
 * @param  edges
 * @returns
 */
export const filterEdgeAndComputeDict = (
    numNodes: number,
    edges: Array<Link>
) => {
    const filteredEdges = [];

    const d: Array<Array<NodeMapLinkEntry>> = new Array(numNodes);
    const h: Record<string, any> = {};
    for (let i = 0; i < numNodes; i++) {
        d[i] = [];
        h[i] = {};
    }
    let k = 0;
    for (const e of edges) {
        if (e.source !== e.target) {
            // not self loops
            const s = Math.min(+e.source, +e.target),
                t = Math.max(+e.source, +e.target);
            if (!Object.hasOwnProperty.call(h[s], t)) {
                // remove dup edges
                d[+e.source].push({ nid: e.target + "", eid: k + "" });
                d[+e.target].push({ nid: e.source + "", eid: k + "" });
                // filteredEdges.push({ ...e, eid: k + "" }); //NOTE 这里并没有保证e中的source,target为string而不是number
                filteredEdges.push({
                    ...e,
                    eid: k + "",
                    source: e.source + "",
                    target: e.target + "",
                });
                k++;
            }
            h[s][t] = true;
        }
    }
    return { edges: filteredEdges, nodeMapLink: d };
};

export const filterEdgesAndComputeDictInMultiGraph = (
    graphArr:
        | Record<
              number,
              {
                  id: Type_GraphId;
                  label: number;
                  nodes: Array<Type_NodeId>;
                  edges: Array<Type_LinkId>;
              }
          >
        | Array<{
              id: Type_GraphId;
              label: number;
              nodes: Array<Type_NodeId>;
              edges: Array<Type_LinkId>;
          }>,
    numAllNodes: number,
    linksRichDict: Record<Type_LinkId, Link>
) => {
    const filteredGraphRecord: Record<
        Type_GraphId,
        {
            nodesRecord: Record<Type_NodeId, Type_GraphId>;
            linksRecord: Record<Type_LinkId, Type_GraphId>;
        }
    > = {};
    const filteredGraphArr: Array<{
        gid: Type_GraphId;
        nodes: Array<Type_NodeId>;
        links: Array<Type_LinkId>;
    }> = [];

    const d: NodeMapLinkEntry[][] = Array.from(
        { length: numAllNodes },
        () => []
    );
    const edges: Array<Link> = [];
    let k = 0;
    for (const graphId in graphArr) {
        const graph = graphArr[graphId];

        const h: Record<string, any> = {};
        for (const nodeId of graph.nodes) {
            h[nodeId] = {};
        }
        const localEdges = [];
        for (const edgeId of graph.edges) {
            const e = linksRichDict[edgeId]; // Assuming you have some way to access edge information
            if (e.source !== e.target) {
                const s = Math.min(+e.source, +e.target);
                const t = Math.max(+e.source, +e.target);
                if (!Object.hasOwnProperty.call(h[s], t)) {
                    d[+e.source].push({ nid: e.target + "", eid: k + "" });
                    d[+e.target].push({ nid: e.source + "", eid: k + "" });
                    localEdges.push({
                        ...e,
                        eid: k + "",
                        source: e.source + "",
                        target: e.target + "",
                    });
                    edges.push({
                        ...e,
                        eid: k + "",
                        source: e.source + "",
                        target: e.target + "",
                    });
                    k++;
                }
                h[s][t] = true;
            }
        }

        filteredGraphRecord[graphId] = {
            nodesRecord: graph.nodes.reduce(
                (acc, cur) => ({ ...acc, [cur + ""]: graphId + "" }),
                {}
            ),
            linksRecord: localEdges.reduce(
                (acc, cur) => ({ ...acc, [cur.eid]: graphId + "" }),
                {}
            ),
        };
        filteredGraphArr.push({
            ...graph,
            gid: graphId + "",
            nodes: graph.nodes.map(String),
            links: localEdges.map((e) => e.eid),
        });
    }

    return {
        filteredGraphRecord,
        filteredGraphArr,
        filteredEdges: edges,
        nodeMapLink: d,
    };
};

/**
 * 计算force-directed layout
 * @param nodes 传入的时候未必有x，y，所以每个点的类型是Node & d3.SimulationNodeDatum
 * @param links
 * @param width
 * @param height
 * @param iter
 * @param radius
 */
export const calcGraphCoords = <
    T extends { id: Type_NodeId } & d3.SimulationNodeDatum,
    U extends { eid: Type_LinkId } & d3.SimulationLinkDatum<T>
>(
    nodes: Array<T>,
    links: Array<U>,
    width = 1000,
    height = 1000,
    iter = 0,
    radius = 2
): (T & d3.SimulationNodeDatum & NodeCoord)[] => {
    // console.log("in calcGraphCoords, nodes:", nodes);
    // console.log("in calcGraphCoords, links:", links);
    const simulation = d3
        .forceSimulation(nodes)
        .force("charge", d3.forceManyBody())
        .force(
            "link",
            d3
                .forceLink<T, U>(links)
                .id((d) => d.id)
                .distance(30)
            // .strength(1)
        )
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collide", d3.forceCollide(radius + 1).iterations(4))
        .stop();
    // simulation.force("link").links(links);

    for (
        let i = 0,
            n =
                iter ||
                Math.ceil(
                    Math.log(simulation.alphaMin()) /
                        Math.log(1 - simulation.alphaDecay())
                );
        i < n;
        ++i
    ) {
        simulation.tick();
        // console.log("in calcGraphCoords, ticking");
    }
    return nodes as (T & d3.SimulationNodeDatum & NodeCoord)[];
    //NOTE 我们确定算完之后有x，y属性
};

export const calcSeparatedMultiGraphCoords = (
    graphArr: Array<{
        gid: Type_GraphId;
        nodes: Array<Type_NodeId>;
        links: Array<{
            eid: Type_LinkId;
            source: Type_NodeId;
            target: Type_NodeId;
        }>;
    }>,
    numNodes: number,
    getEachWidth: (graphId: Type_GraphId) => number = () => 100,
    getEachHeight: (graphId: Type_GraphId) => number = () => 100,
    iter = 0,
    radius = 2
): Array<d3.SimulationNodeDatum & NodeCoord> => {
    const coords = new Array(
        numNodes || graphArr.reduce((acc, cur) => acc + cur.nodes.length, 0)
    );
    for (const g of graphArr) {
        // console.log(
        //     "in graphUtils, in calcSeparatedMultiGraphCoords, now processing",
        //     g,
        //     "typeof g.nodes[0], typeof g.links[0].eid",
        //     typeof g.nodes[0],
        //     typeof g.links[0].eid
        // );
        const tmpCoords = calcGraphCoords(
            g.nodes.map((d) => ({
                id: d,
            })),
            g.links,
            getEachWidth(g.gid),
            getEachHeight(g.gid),
            iter,
            radius
        );
        for (const c of tmpCoords) {
            coords.push(c);
        }
    }
    return coords;
};

/**
 * rescaleCoords by 2d ranges
 * @param coords ! NOTE: use origin coords can avoid latent bugs, \
 * because when component activate/deactivate, the width and height of views will change\
 * so calculated xRange and yRange may be negative, incur unknown bugs.
 * @param xRange if number, [0, xRange], else xRange itself is a Array or Iterable
 * @param yRange if number, [0, yRange], else yRange itself is a Array or Iterable
 * @param getXFunc how to get x coord of each element?
 * @param getYFunc how to get y coord of each element?
 * @returns
 */
export const rescaleCoords = <
    T extends { id?: Type_NodeId; x: number; y: number },
    U extends [number, number] = [number, number]
>(
    coords: Array<T | U>,
    xRange: number | Iterable<number>, // newWidth,
    yRange: number | Iterable<number>, // newHeight,
    getXFunc: (d: T | U) => number = (d: T | U) => ("x" in d ? d.x : d[0]),
    getYFunc: (d: T | U) => number = (d: T | U) => ("y" in d ? d.y : d[1]),
    debugViewName = ""
): Array<T | { id: Type_NodeId; x: number; y: number }> => {
    //
    // console.log(
    //     "in graphUtil, in rescale, viewName",
    //     debugViewName,
    //     "coords.length: ",
    //     coords.length,
    //     "maybe coords[0]",
    //     coords.length > 0 ? coords[0] : undefined,
    //     "\nxRange:",
    //     xRange,
    //     "\nyRange",
    //     yRange
    // );
    const sx = d3
        .scaleLinear()
        .domain(d3.extent(coords.map((d) => getXFunc(d))) as [number, number])
        .range(typeof xRange === "number" ? [0, xRange] : xRange);
    const sy = d3
        .scaleLinear()
        .domain(d3.extent(coords.map((d) => getYFunc(d))) as [number, number])
        .range(typeof yRange === "number" ? [0, yRange] : yRange);
    return coords.map((c, i) =>
        Array.isArray(c)
            ? {
                  x: sx(getXFunc(c)),
                  y: sy(getYFunc(c)),
                  id: i + "",
              }
            : {
                  ...(c as T),
                  x: sx(getXFunc(c)),
                  y: sy(getYFunc(c)),
                  id: Object.hasOwn(c, "id") ? c.id : i + "",
              }
    );
};

/**
 * rescale 2d polar coordinates, and can be used in piecewise ones
 * @param coords 坐标数组
 * @param getRadiusFunc 如何通过某个点得到其半径 \ // NOTE 每次应该用最初值.因为不是线性的，每次rescale用上一次的结果,会导致越来越集中。
 * @param getAngleFunc 如何通过某个点得到其极角
 * @param getRadiusRangeByHop
 * @param getAngleRange
 * @param getHop 可以针对不同半径分段处理。此函数得到某个点所在分段。
 * @returns
 */
export const rescalePolarCoords = <
    T extends { id?: Type_NodeId; angle: number; radius: number; hop?: number }
>(
    coords: Array<T>,
    getRadiusFunc: (d: T) => number = (d: T) => d.radius,
    // "radius" in d ? d.radius : d[0],
    // Object.hasOwn(d, "radius") ? (d as T).radius : (d as U)[0],
    getAngleFunc: (d: T) => number = (d: T) => d.angle,
    // "angle" in d ? d.radius : d[0],
    // Object.hasOwn(d, "radius") ? (d as T).radius : (d as U)[0],
    getRadiusRangeByHop: (hop: number, ...args: any[]) => [number, number],
    getAngleRange: (...args: any[]) => [number, number] = () => [0, Math.PI],
    getHop: (d: T) => number | undefined = () => undefined
): Array<
    | T
    | {
          id: Type_NodeId;
          angle: number;
          radius: number;
          hop: number | undefined;
      }
> => {
    console.log(
        "in rescalePolarCoords, getRadiusFunc,getAngleFunc, getRadiusRangeByHop(1),getAngleRange() ",
        getRadiusFunc,
        getAngleFunc,
        getRadiusRangeByHop(1),
        getAngleRange()
    );
    const scaleRadiusByHop = (hop: number | undefined) =>
        typeof hop === "number"
            ? d3
                  .scaleRadial()
                  .domain(
                      // getRadiusDomain(hop, getRadiusFunc)
                      d3.extent(
                          coords
                              .filter((d) => getHop(d) === hop)
                              .map((d) => getRadiusFunc(d))
                      ) as [number, number]
                  )
                  .range(getRadiusRangeByHop(hop))
            : d3 // 无hop
                  .scaleRadial()
                  .domain(
                      d3.extent(coords.map((d) => getRadiusFunc(d))) as [
                          number,
                          number
                      ]
                  );
    const scaleAngle = () =>
        d3
            .scaleLinear()
            .domain(
                d3.extent(coords.map((d) => getAngleFunc(d))) as [
                    number,
                    number
                ] //统一,不分段
            )
            .range(getAngleRange());

    return coords.map((c, i) =>
        // Array.isArray(c)
        //     ? {
        //           id: i + "",
        //           radius: scaleRadiusByHop(getHop(c as U))(
        //               getRadiusFunc(c as U)
        //           ),
        //           angle: scaleAngle()(getAngleFunc(c as U)),
        //           hop: getHop(c as U),
        //       }
        //     :
        ({
            ...(c as T),
            radius: scaleRadiusByHop(getHop(c as T))(getRadiusFunc(c as T)),
            angle: scaleAngle()(getAngleFunc(c as T)),
            id: "id" in c ? c.id : i + "",
        })
    );
};

export const calcTsne = (
    data: Array<Array<number>>,
    iterCount = 2,
    perplexity = 30,
    epsilon = 10
): [number, number][] => {
    const tsneObj = new tsnejs.tSNE({ perplexity, epsilon }); // LINK public/workers/tsne.js/#tsnejs
    tsneObj.initDataRaw(data);
    for (let k = 0; k < iterCount; k++) {
        tsneObj.step();
    }
    return tsneObj.getSolution();
};

/**
 * 计算src中每一个向量到tgt所有向量的平均距离(cosine)
 * @param srcDict
 * @param tgtDict
 * @param embAll
 * @param exclusive 是否在target中排除src有的
 */
export const calcVectorDist = (
    srcDictOrArr: Record<Type_NodeId, any> | Array<Type_NodeId>,
    tgtDictOrArr: Record<Type_NodeId, any> | Array<Type_NodeId>,
    embAll: number[][],
    exclusive = true
) => {
    const srcIndexes = Array.isArray(srcDictOrArr)
        ? srcDictOrArr
        : Object.keys(srcDictOrArr).filter((d) => Boolean(srcDictOrArr[d]));
    const isIdInSrc = Array.isArray(srcDictOrArr)
        ? (id: Type_NodeId) => srcDictOrArr.includes(id)
        : (id: Type_NodeId) => srcDictOrArr[id];

    const tgtIndexes = Array.isArray(tgtDictOrArr)
        ? tgtDictOrArr.filter((id) => (exclusive ? !isIdInSrc(id) : true))
        : Object.keys(tgtDictOrArr).filter(
              (id) =>
                  Boolean(tgtDictOrArr[id]) &&
                  (exclusive ? !isIdInSrc(id) : true)
          );
    return srcIndexes.map((srcIndex) => ({
        id: srcIndex,
        dist:
            tgtIndexes.reduce(
                (acc, cur) =>
                    acc + getCosineDistance(embAll[+srcIndex], embAll[+cur]),
                0
            ) / tgtIndexes.length,
    }));
};

export const calcRank3 = (
    algo: "single" | "average" | "center" | undefined,
    nodeOrNodes: string | number | Record<string, any> | Array<string | number>,
    embAll1: Array<Array<number>>,
    embAll2: Array<Array<number>>
) => {
    console.log("in calcRank3, got nodeOrNodes", nodeOrNodes);
    let nodesArr: Array<number>;
    if (typeof nodeOrNodes === "string") {
        nodesArr = [Number.parseInt(nodeOrNodes)]; // 单点
    } else if (typeof nodeOrNodes === "number") {
        nodesArr = [nodeOrNodes];
    } else if (Array.isArray(nodeOrNodes) || typeof nodeOrNodes === "object") {
        nodesArr = Array.isArray(nodeOrNodes)
            ? nodeOrNodes
            : Object.keys(nodeOrNodes || {}).map((d) => Number.parseInt(d)); //NOTE 此parseInt函数有坑
        if (nodesArr.length === 0) throw new Error("no nodes selected!");
    } else {
        throw new Error("no nodes selected!");
    }

    console.log("in calcRank3, nodesArr", nodesArr);
    if (algo === "single") {
        const nodeEmb1 = embAll1[nodesArr[0]];
        const nodeEmb2 = embAll2[nodesArr[0]];
        const distances1 = embAll1.map((e) => getCosineDistance(e, nodeEmb1)); // LINK public/workers/distance.js/#getCosineDistance
        const distances2 = embAll2.map((e) => getCosineDistance(e, nodeEmb2)); // LINK public/workers/distance.js/#getCosineDistance

        return calcRank(distances1, distances2); // LINK  public/workers/distance.js/#calcRank
    } else if (algo === "average") {
        const nodesEmbs1 = nodesArr.map((id) => embAll1[id]);
        const nodesEmbs2 = nodesArr.map((id) => embAll2[id]);

        const distances1 = embAll1.map(
            (e) =>
                nodesEmbs1.reduce(
                    (acc, cur) => acc + getCosineDistance(cur, e),
                    0
                ) / nodesEmbs1.length // LINK public/workers/distance.js/#getCosineDistance
        );

        const distances2 = embAll2.map(
            (e) =>
                nodesEmbs2.reduce(
                    (acc, cur) => acc + getCosineDistance(cur, e),
                    0
                ) / nodesEmbs2.length // LINK public/workers/distance.js/#getCosineDistance
        );

        return calcRank(distances1, distances2); // LINK  public/workers/distance.js/#calcRank
    } else if (algo === "center") {
        const nodesEmbs1 = nodesArr.map((id) => embAll1[id]);
        const nodesEmbs2 = nodesArr.map((id) => embAll2[id]);

        const center1 = nodesEmbs1
            .reduce(
                (acc, cur) => cur.map((dim, i) => dim + acc[i]),
                new Array(nodesEmbs1[0].length).fill(0)
            ) // element-wise add
            .map((dim) => dim / nodesEmbs1.length);

        const center2 = nodesEmbs2
            .reduce(
                (acc, cur) => cur.map((dim, i) => dim + acc[i]),
                new Array(nodesEmbs2[0].length).fill(0)
            ) // element-wise add
            .map((dim) => dim / nodesEmbs2.length);

        const distances1 = embAll1.map((e) => getCosineDistance(e, center1)); // LINK public/workers/distance.js/#getCosineDistance
        const distances2 = embAll2.map((e) => getCosineDistance(e, center2)); // LINK public/workers/distance.js/#getCosineDistance
        const ret = calcRank(distances1, distances2); // LINK  public/workers/distance.js/#calcRank
        return ret;
    } else {
        throw Error("unsupported rank calc algo");
    }
};

export const calcPolar = (
    topoAlgo: "hamming" | "jaccard" | "shortest path" | undefined,
    embAlgo: "center" | "average" | "single" | undefined,
    nodeOrNodes: string | number | Record<string, any> | Array<string | number>,
    embAll1: Array<Array<number>>,
    embAll2: Array<Array<number>>,
    nodeMapLink: any,
    hops: number,
    neighborMasksByHop: Array<Array<string>>
) => {
    if (
        !Array.isArray(neighborMasksByHop) ||
        hops > neighborMasksByHop.length
    ) {
        throw new Error("illegal hops or masks");
    }

    console.log("in calcPolar, got nodeOrNodes", nodeOrNodes);
    let nodesArr: Array<number>;
    if (typeof nodeOrNodes === "string") {
        nodesArr = [Number.parseInt(nodeOrNodes)]; // 单点
    } else if (typeof nodeOrNodes === "number") {
        nodesArr = [nodeOrNodes];
    } else if (Array.isArray(nodeOrNodes) || typeof nodeOrNodes === "object") {
        nodesArr = Array.isArray(nodeOrNodes)
            ? nodeOrNodes
            : Object.keys(nodeOrNodes || {}).map((d) => Number.parseInt(d));
        if (nodesArr.length === 0) throw new Error("no nodes selected!");
    } else {
        throw new Error("no nodes selected!");
    }
    console.log("in calcPolar, nodesArr", nodesArr);

    let ret: PolarDatum[] = [], //[{id,hop,embDiff,topoDist}, {id,hop,embDiff,topoDist}, ...]
        curRet: PolarDatum[] = [];

    let neighborBS = new BitSet(nodesArr); //所选点（们）的1阶neighborMask，代表了所选点（们）的邻居点
    nodesArr.forEach((d) => {
        neighborBS = neighborBS.or(new BitSet("0x" + neighborMasksByHop[0][d]));
    });
    nodesArr.forEach((d) => neighborBS.set(d, 0)); // NOTE not include self

    let neighborArr = neighborBS.toArray();
    const nodesEmbs1 = nodesArr.map((d) => embAll1[d]); //[[]],也有可能[[],[],[]]
    const nodesEmbs2 = nodesArr.map((d) => embAll2[d]);
    // console.log("in calcPolar , in Universal, nodesEmbs1", nodesEmbs1);
    for (let j = 0; j < hops; ++j) {
        //计算embDiffs
        let distances1: number[], distances2: number[];
        if (embAlgo === "single") {
            distances1 = neighborArr.map(
                (d) => getCosineDistance(embAll1[d], nodesEmbs1[0]) // LINK public/workers/distance.js/#getCosineDistance
            );
            distances2 = neighborArr.map(
                (d) => getCosineDistance(embAll2[d], nodesEmbs2[0]) // LINK public/workers/distance.js/#getCosineDistance
            );
        } else if (embAlgo === "average") {
            distances1 = neighborArr.map(
                (d) =>
                    nodesEmbs1.reduce(
                        (acc, cur) => acc + getCosineDistance(cur, embAll1[d]), // LINK public/workers/distance.js/#getCosineDistance
                        0
                    ) / nodesEmbs1.length
            );

            distances2 = neighborArr.map(
                (d) =>
                    nodesEmbs2.reduce(
                        (acc, cur) => acc + getCosineDistance(cur, embAll2[d]), // LINK public/workers/distance.js/#getCosineDistance
                        0
                    ) / nodesEmbs2.length
            );
        } else if (embAlgo === "center") {
            const center1 = nodesEmbs1
                .reduce(
                    (acc, cur) => cur.map((dim, i) => dim + acc[i]),
                    new Array(nodesEmbs1[0].length).fill(0)
                ) // element-wise add
                .map((dim) => dim / nodesEmbs1.length);

            const center2 = nodesEmbs2
                .reduce(
                    (acc, cur) => cur.map((dim, i) => dim + acc[i]),
                    new Array(nodesEmbs2[0].length).fill(0)
                ) // element-wise add
                .map((dim) => dim / nodesEmbs2.length);

            distances1 = neighborArr.map(
                (d) => getCosineDistance(embAll1[d], center1) // LINK public/workers/distance.js/#getCosineDistance
            );
            distances2 = neighborArr.map(
                (d) => getCosineDistance(embAll2[d], center2) // LINK public/workers/distance.js/#getCosineDistance
            );
        } else {
            throw new Error("unsupported emb diff algorithm");
        }
        curRet = distances1.map(
            (d, i) =>
                ({
                    id: neighborArr[i] + "",
                    embDiff: Math.abs(d - distances2[i]), //REVIEW how to calc diff? plain subtraction? abs?
                    hop: j, //ANCHOR - definition related: hop从0开始的！
                    // topoDist:
                } as PolarDatum)
        );

        //计算topoDists
        let topoDists;
        // console.log("in Universal, j", j);
        const curNeighborMasks =
            j === 0
                ? neighborArr.map(
                      (d) => new BitSet("0x" + neighborMasksByHop[0][d])
                  )
                : neighborArr.map((d) =>
                      new BitSet("0x" + neighborMasksByHop[j][d]).xor(
                          new BitSet("0x" + neighborMasksByHop[j - 1][d])
                      )
                  );
        neighborArr.forEach((d, i) => curNeighborMasks[i].set(d, 0)); //先排除本身
        if (topoAlgo === "shortest path") {
            const paths = neighborArr.map(
                (d) => computeShortestPath(d, new BitSet(nodesArr), nodeMapLink) // LINK  public/workers/distance.js/#computeShortestPath
                //应当让遍历的点作为第一个参数，所选点作为bitset作为第二个参数，下面的与此统一
            );
            topoDists = paths.map((d) => (Array.isArray(d) ? d.length : 0)); //REVIEW 0 or -1?
        } else if (topoAlgo === "jaccard") {
            topoDists = neighborArr.map(
                (d, i) =>
                    getJaccardDistance(
                        curNeighborMasks[i], //所选点每个邻居点的neighborMask(不包括本身)
                        neighborBS //所选点的邻居点（不包括本身）共同组成的mask,即所选点（们）的neighborMask(不包括本身)
                    ) // LINK  public/workers/distance.js/#getJaccardDistance
            );
        } else if (topoAlgo === "hamming") {
            topoDists = neighborArr.map(
                (d, i) =>
                    getHammingDistance(
                        curNeighborMasks[i], //所选点每个邻居点的neighborMask(不包括本身)
                        neighborBS //所选点的邻居点（不包括本身）共同组成的mask,即所选点（们）的neighborMask(不包括本身)
                    ) // LINK  public/workers/distance.js/#getHammingDistance
            );
        } else {
            throw new Error("unsupported topo dist algorithm");
        }
        topoDists.forEach((d, i) => {
            curRet[i].topoDist = d;
        });
        ret = [...ret, ...curRet];

        if (j + 1 < hops) {
            let bsCur = new BitSet(); //这里我们选用bsCur而不是neighborBS去更新每次，是因为neighborBS作为最初NOI的neighborMask，在每次计算中是有用的
            nodesArr.forEach((d) => {
                bsCur = bsCur.or(
                    new BitSet("0x" + neighborMasksByHop[j + 1][d])
                );
            });
            let bsPrev = new BitSet();
            nodesArr.forEach((d) => {
                bsPrev = bsPrev.or(new BitSet("0x" + neighborMasksByHop[j][d]));
            });
            bsCur = bsCur.xor(bsPrev);
            nodesArr.forEach((d) => bsCur.set(d, 0));
            neighborArr = bsCur.toArray();
        }
    }
    return ret;
};

export const calcHexbinClusters = <
    T extends { id: Type_NodeId; x: number; y: number }
>(
    points: Array<T>,
    extentFunc: () => [[number, number], [number, number]], //function
    getXFunc: (d: T) => number = (d) => d.x,
    getYFunc: (d: T) => number = (d) => d.y,
    radius: number, //radius is static
    getCountAttrOfT: (d: T) => string | number | symbol,
    debugInfo: string = ""
): {
    hb: Hexbin<T>;
    pointCluster: Record<Type_NodeId, Type_ClusterId>;
    clusters: {
        id: Type_ClusterId;
        pointIds: Type_NodeId[];
        pointCenter: { x: number; y: number }; //absolute
        hexCenter: { x: number; y: number };
        count: Record<string | number | symbol, number>;
    }[];
} => {
    console.log("in calcHexbinClusters, got points", points);
    // console.log("in calcHexbinClusters, got extent", extentFunc());

    //NOTE 此函数不应该再用scale，不然会很复杂。直接什么坐标就是什么坐标
    // const sx = d3
    //     .scaleLinear()
    //     .domain(d3.extent(points.map((d) => getXFunc(d))) as [number, number])
    //     .range([extentFunc()[0][0], extentFunc()[1][0]]);
    // const sy = d3
    //     .scaleLinear()
    //     .domain(d3.extent(points.map((d) => getYFunc(d))) as [number, number])
    //     .range([extentFunc()[0][1], extentFunc()[1][1]]);

    const hb = hexbin<T>()
        .extent(extentFunc())
        .radius(radius)
        .x((d) => getXFunc(d))
        .y((d) => getYFunc(d));
    // .x((d) => sx(getXFunc(d)))
    // .y((d) => sy(getYFunc(d)));

    // console.log(
    //     "in calcHexbinClusters, hb.centers().length ",
    //     hb.centers().length
    // );

    // console.log("in calcHexbinClusters debugInfo", debugInfo);
    const ret1 = hb(points);

    console.log(
        "in calcHexbinClusters, hb(points).length",
        ret1.length,
        // "\nhb(points):",
        // ret1,
        "\nmaybe hb(points)[0]",
        ret1[0],
        "\nmaybe hb.mesh()[0:50]",
        hb.mesh().slice(0, 50)
    );
    return {
        hb: hb,

        pointCluster: ret1.reduce(
            (acc, cur, curI) => ({
                ...acc,
                ...cur.reduce(
                    (inAcc, inCur) => ({
                        ...inAcc,
                        [inCur.id]: curI + "",
                    }),
                    {}
                ),
            }),
            {}
        ),
        clusters: ret1.map((arrAndXY, i) => ({
            //这个arrAndXY是一个数组，但是加了xy属性（hex中心）
            //数组各元素为落在这个hex的原始点对象(注意这里面的坐标仍然是原坐标)
            id: i + "",
            pointIds: arrAndXY.map((obj) => obj.id),
            pointCenter: arrAndXY.reduce(
                (acc, cur) => ({
                    // x: acc.x + sx(getXFunc(cur)) / arrAndXY.length,
                    // y: acc.y + sy(getYFunc(cur)) / arrAndXY.length,
                    x: acc.x + getXFunc(cur) / arrAndXY.length,
                    y: acc.y + getYFunc(cur) / arrAndXY.length,
                }),
                { x: 0, y: 0 }
            ),
            hexCenter: { x: arrAndXY.x, y: arrAndXY.y },
            count: arrAndXY.reduce((acc, cur) => {
                const value = getCountAttrOfT(cur);
                acc[value] = (acc[value] || 0) + 1;
                return acc;
            }, {} as Record<ReturnType<typeof getCountAttrOfT>, number>),
        })),
    };
};

export const calcAggregatedLinks = (
    clusterPoints: Type_NodeId[][], //一层为索引，二层为各栅格中包含的node的ids
    linkArr: Array<Link>,
    getClusterIndexOfNode: (d: Type_NodeId) => Type_ClusterId
) => {
    const aggregatedLinkArr: AggregatedLink[] = [];
    const clusterAdjMatrix: Link[][][] = clusterPoints.map(() =>
        clusterPoints.map(() => [])
    );
    console.log("in calcAggregatedLinks, linkArr", linkArr);

    linkArr.forEach((link) => {
        const srcCluster = getClusterIndexOfNode(link.source);
        const tgtCluster = getClusterIndexOfNode(link.target);
        if (srcCluster && tgtCluster)
            clusterAdjMatrix[+srcCluster][+tgtCluster].push(link);
    });

    let k = 0;
    for (let i = 0; i < clusterPoints.length; i++) {
        for (let j = 0; j < clusterPoints.length; j++) {
            if (clusterAdjMatrix[i][j].length > 0) {
                aggregatedLinkArr.push({
                    aeid: k++ + "",
                    source: i + "",
                    target: j + "",
                    baseLinks: clusterAdjMatrix[i][j],
                });
            }
        }
    }
    return aggregatedLinkArr;
};
