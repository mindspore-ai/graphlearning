"use strict"; //NOTE 记得导出每个函数
const distance = {};
// importScripts("d3.js");
// importScripts("https://unpkg.com/vue@3/dist/vue.esm-browser.js");
// import * as d3 from "d3";
// import { isReactive } from "vue";
// const { ref, reactive, toRaw } = require("https://unpkg.com/vue@3/dist/vue.esm-browser.js");

const sim2dist = d3.scaleLinear().domain([-1, 1]).range([1, 0]);

// ANCHOR[id=getCosineDistance]
const getCosineDistance = (u, v) => {
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

//ANCHOR[id=getJaccardDistance]
const getJaccardDistance = (mask1, mask2) => {
    // console.log(`in getJaccardDistance, mask1:${mask1},mask2:${mask2}`);
    // console.log(`in getJaccardDistance, mask1 instanceof BitSet:${mask1 instanceof BitSet}`);//false!
    // console.log(`in getJaccardDistance,  isReacitve(mask1):${isReactive(mask1)}`);
    // console.log(`in getJaccardDistance, mask1.__proto__ ${mask1.__proto__}`);
    // Jaccard distance

    const bs1 = typeof mask1 === "string" ? new BitSet("0x" + mask1) : mask1;
    const bs2 = typeof mask2 === "string" ? new BitSet("0x" + mask2) : mask2;
    const intersection = bs1.and(bs2).cardinality();
    const union = bs1.or(bs2).cardinality();
    return union === 0 ? 0 : 1 - intersection / union;
};

//ANCHOR[id=getHammingDistance]
const getHammingDistance = (mask1, mask2) => {
    const bs1 = typeof mask1 === "string" ? new BitSet("0x" + mask1) : mask1;
    const bs2 = typeof mask2 === "string" ? new BitSet("0x" + mask2) : mask2;
    // return mask1.xor(mask2).cardinality(); //bitset中arr中元素的个数
    return bs1.xor(bs2).cardinality(); //bitset中arr中元素的个数
};

// ANCHOR[id=calcRank]
const calcRank = (distances1, distances2) => {
    const getRanks = (distances) => {
        const sorted = distances.sort((a, b) => a.d - b.d);
        return sorted
            .map((x, i) => ({ ...x, r: i }))
            .sort((a, b) => a.id - b.id);
    };
    const ranks1 = getRanks(distances1.map((d, i) => ({ id: i, d }))); //the d is named as d
    const ranks2 = getRanks(distances2.map((d, i) => ({ id: i, d })));

    //ANCHOR - definition related
    const ranksMerge = ranks1.map((r1, i) => ({
        id: i,
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

/** ANCHOR[id=computeShortestPath]
 * compute the shortest path between a node and a node group // BFS
 * @param {*} node
 * @param {*} targetBS
 * @param {*} linkDict
 * @returns
 */
function computeShortestPath(node, targetBS, linkDict) {
    if (targetBS.get(node) === 1) return -1;
    const q = [node];
    const prev = new Array(linkDict.length).fill(-1);
    const visited = { [node]: true };
    while (q.length > 0) {
        let cur = q.shift();
        for (let e of linkDict[cur]) {
            if (targetBS.get(e.nid) === 1) {
                // shortest path found
                const sp = [e.nid, cur];
                while (prev[cur] !== -1) {
                    cur = prev[cur];
                    sp.push(cur);
                }
                return sp;
            }
            if (!visited[e.nid]) {
                visited[e.nid] = true;
                prev[e.nid] = cur;
                q.push(e.nid);
            }
        }
    }
    return -1;
}
// NOTE 记得导出！
distance.getCosineDistance = getCosineDistance;
distance.calcRank = calcRank;
distance.computeShortestPath = computeShortestPath;
distance.getHammingDistance = getHammingDistance;
distance.getJaccardDistance = getJaccardDistance;
if (typeof module != "undefined") module.exports = distance;

// export { getCosineDistance, getHammingDistance, calcRank, getJaccardDistance };
