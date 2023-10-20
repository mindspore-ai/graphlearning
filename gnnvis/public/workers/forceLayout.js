"use strict"; //NOTE 记得导出每个函数
const forceLayoutWorker = {};
const calcGraphCoords = (
    nodes,
    links,
    width = 1000,
    height = 1000,
    iter = 0,
    radius = 2
) => {
    // console.log("in calcGraphCoords, nodes:", nodes);
    // console.log("in calcGraphCoords, links:", links);
    const simulation = d3
        .forceSimulation(nodes)
        .force("charge", d3.forceManyBody())
        .force(
            "link",
            d3
                .forceLink(links)
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
    return nodes;
    //NOTE 我们确定算完之后有x，y属性
};

forceLayoutWorker.calcGraphCoords = calcGraphCoords;
if (typeof module != "undefined") module.exports = forceLayoutWorker;
