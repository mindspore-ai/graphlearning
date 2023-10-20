import type { Type_SymbolName } from "@/types/types";

/**
 * 获取一个base64图片的尺寸
 * @param base64Data
 */
export const getImageDimensionsFromBase64Async = (base64Data: string) => {
    return new Promise<{ width: number; height: number }>((resolve, reject) => {
        if (!base64Data)
            reject(new Error(`failed to load img ->${base64Data}<-`));

        const img = new Image();
        img.src = base64Data;

        img.onload = function () {
            const width = img.width;
            const height = img.height;
            resolve({ width, height });
        };

        img.onerror = function () {
            reject(new Error(`failed to load img ->${base64Data}<-`));
        };
    });
};

/**
 * 以中心为基准的一个圆，直接返回path字符串
 * @param R 半径
 */
export const circlePath = (R: number) =>
    `M0 ${R} A${R} ${R} 0 1 1 ${0} ${-R} a${R} ${R} 0 0 1 ${0} ${2 * R} z`;
/**
 * 以中心为基准的左半圆，直接返回path字符串
 * @param R 半径
 */
export const leftHalfCirclePath = (R: number) =>
    `M${0} ${R} A${R} ${R} 0 1 1 ${0} ${-R} v${2 * R} z`;
/**
 * 以中心为基准的右半圆，直接返回path字符串
 * @param R 半径
 */
export const rightHalfCirclePath = (R: number) =>
    `M${0} ${R} A${R} ${R} 0 1 0 ${0} ${-R} v${2 * R} z`;
/**
 * 以中心为基准的扇形圆，以右边的比例计算，直接返回path字符串
 * @param R 半径
 * @param percent 右边填充的占比，0-1。如果给了
 */
export const rightPercentCirclePath = (R: number, percent: number) => {
    const halfRadian =
        percent < 0
            ? 0
            : percent > 100
            ? 1
            : ((percent > 1 ? percent / 100 : percent) / 2) * 2 * Math.PI;
    return `M${0} ${0} L${R * Math.cos(halfRadian)} ${
        R * Math.sin(halfRadian)
    } A${R} ${R} 0 ${
        percent > 0.5 ? 1 : 0 //优弧or劣弧
    } 0 ${R * Math.cos(halfRadian)} ${-R * Math.sin(halfRadian)} L0 0 z`;
};
/**
 * 以中心为基准的扇形圆，以左边的比例计算，直接返回path字符串
 * @param R 半径
 * @param percent 右边填充的占比，0-1。如果给了
 */
export const leftPercentCirclePath = (R: number, percent: number) => {
    const halfRadian =
        percent < 0
            ? 0
            : percent > 100
            ? 1
            : ((percent > 1 ? percent / 100 : percent) / 2) * 2 * Math.PI;
    return `M${0} ${0} L${-R * Math.cos(halfRadian)} ${
        R * Math.sin(halfRadian)
    } A${R} ${R} 0 ${
        percent > 0.5 ? 1 : 0 //优弧or劣弧
    } 1 ${-R * Math.cos(halfRadian)} ${-R * Math.sin(halfRadian)} L0 0 z`;
};

/**
 * 以中心为基准的正三角形，直接返回path字符串
 * @param R 外接圆半径
 */
export const trianglePath = (R: number) =>
    `M 0 ${-R} L ${(-R * Math.sqrt(3)) / 2} ${R / 2} L ${
        (R * Math.sqrt(3)) / 2
    } ${R / 2} L ${0} ${-R} z`;
/**
 * 以正三角形中心为基准的左半正三角形，直接返回path字符串
 * @param R 外接圆半径
 */
export const leftHalfTrianglePath = (R: number) =>
    `M${0} ${-R} V${R / 2} L${-(R * Math.sqrt(3)) / 2} ${R / 2} L${0} ${-R} z`;
/**
 * 以正三角形中心为基准的右半正三角形，直接返回path字符串
 * @param R 外接圆半径
 */
export const rightHalfTrianglePath = (R: number) =>
    `M${0} ${-R} V${R / 2} L${(R * Math.sqrt(3)) / 2} ${R / 2} L${0} ${-R} z`;

/**
 * 以中心为基准的正方形，直接返回path字符串
 * @param R 外接圆半径
 */
export const rectPath = (R: number) => {
    const L = R / Math.sqrt(2); //L是边长的一半
    return `M${-L} ${-L} L ${L} ${-L} L ${L} ${L} L ${-L} ${L} L ${-L} ${-L} z`;
};
/**
 * 以正方形中心为基准的左半正方形，直接返回path字符串
 * @param R 外接圆半径
 */
export const leftHalfRectPath = (R: number) => {
    const L = R / Math.sqrt(2); //L是长边长的一半
    return `M${0} ${-L} L ${0} ${L} L ${-L} ${L} L ${-L} ${-L} L ${0} ${-L} z`;
};
/**
 * 以正方形中心为基准的右半正方形，直接返回path字符串
 * @param R 外接圆半径
 */
export const rightHalfRectPath = (R: number) => {
    const L = R / Math.sqrt(2); //L是长边长的一半
    return `M${0} ${-L} L ${0} ${L} L ${L} ${L} L ${L} ${-L} L ${0} ${-L} z`;
};

/**
 * 以中心为基准的十字，直接返回path字符串
 * @param R 外切圆半径
 */
export const crossPath = (R: number) => {
    const L = (R / 3) * 2;
    return `M${
        L / 2
    } ${-R} v${L} h${L} v${L} h${-L} v${L} h${-L} v${-L} h${-L} v${-L} h${L} v${-L} h${L} z`;
};
/**
 * 以十字中心为基准的左半十字，直接返回path字符串
 * @param R 外切圆半径
 */
export const leftHalfCrossPath = (R: number) => {
    const L = (R / 3) * 2;
    return `M${0} ${-R} V${R} H${-L / 2} V${L / 2} H${-R} V${-L / 2} H${
        -L / 2
    } V${-R} H${0} z`;
};
/**
 * 以十字中心为基准的右半十字，直接返回path字符串
 * @param R 外切圆半径
 */
export const rightHalfCrossPath = (R: number) => {
    const L = (R / 3) * 2;
    return `M${0} ${-R} V${R} H${L / 2} V${L / 2} H${R} V${-L / 2} H${
        L / 2
    } V${-R} H${0} z`;
};

/**
 * 以中心为基准的菱形，直接返回path字符串
 * @param R 长轴外接圆半径
 */
export const diamondPath = (R: number) => {
    return `M${0} ${-R} L${-R / Math.sqrt(3)} ${0} L0 ${R} L${
        R / Math.sqrt(3)
    } ${0} L0 ${-R} z`;
};
/**
 * 以菱形中心为基准的左半菱形，直接返回path字符串
 * @param R 长轴外接圆半径
 */
export const leftHalfDiamondPath = (R: number) => {
    return `M${0} ${-R} L${-R / Math.sqrt(3)} ${0} L0 ${R} L0 ${-R} z`;
};
/**
 * 以菱形中心为基准的右半菱形，直接返回path字符串
 * @param R 长轴外接圆半径
 */
export const rightHalfDiamondPath = (R: number) => {
    return `M${0} ${-R} L${R / Math.sqrt(3)} ${0} L0 ${R} L0 ${-R} z`;
};

/**
 * 以中心为基准的Y形，直接返回path字符串
 * @param R 外切圆半径
 */
export const wyePath = (R: number) => {
    const a = R / (1 + 1 / Math.sqrt(12));
    const triR = a / Math.sqrt(3);
    return `M0 ${-triR} L${(a * Math.sqrt(3)) / 2} ${-(a / 2 + triR)} l${
        a / 2
    } ${(a / 2) * Math.sqrt(3)}  L${(triR / 2) * Math.sqrt(3)} ${
        triR / 2
    } v${a} h${-a} v${-a} l${(-a / 2) * Math.sqrt(3)} ${-a / 2} L${
        (-a * Math.sqrt(3)) / 2
    } ${-(a / 2 + triR)} L${0} ${-triR} z`;
};
/**
 * 以Y形中心为基准的左半Y形，直接返回path字符串
 * @param R 外切圆半径
 */
export const leftHalfWyePath = (R: number) => {
    const a = R / (1 + 1 / Math.sqrt(12));
    const triR = a / Math.sqrt(3);
    return `M0 ${-triR} L${(-a * Math.sqrt(3)) / 2} ${-(a / 2 + triR)} l${
        -a / 2
    } ${(a / 2) * Math.sqrt(3)}  L${(-triR / 2) * Math.sqrt(3)} ${
        triR / 2
    } v${a} h${a / 2} V${-triR}z`;
};
/**
 * 以Y形中心为基准的右半Y形，直接返回path字符串
 * @param R 外切圆半径
 */
export const rightHalfWyePath = (R: number) => {
    const a = R / (1 + 1 / Math.sqrt(12));
    const triR = a / Math.sqrt(3);
    return `M0 ${-triR} L${(a * Math.sqrt(3)) / 2} ${-(a / 2 + triR)} l${
        a / 2
    } ${(a / 2) * Math.sqrt(3)}  L${(triR / 2) * Math.sqrt(3)} ${
        triR / 2
    } v${a} h${-a / 2} V${-triR}z`;
};

/**
 * 以中心为基准的五角星，直接返回path字符串
 * @param R 外接圆半径
 */
export const starPath = (R: number) => {
    const h =
        R /
        (1 / Math.tan((36 / 180) * Math.PI) +
            1 / Math.tan((18 / 180) * Math.PI));
    const r = h / Math.sin((36 / 180) * Math.PI); //内部正五边形的外接圆半径
    return `M0 ${-R} L${h} ${-r * Math.cos((36 / 180) * Math.PI)} H${
        R * Math.sin((72 / 180) * Math.PI)
    } L${r * Math.cos((18 / 180) * Math.PI)} ${
        r * Math.sin((18 / 180) * Math.PI)
    } L${R * Math.sin((36 / 180) * Math.PI)} ${
        R * Math.cos((36 / 180) * Math.PI)
    } L${0} ${r} L${-R * Math.sin((36 / 180) * Math.PI)} ${
        R * Math.cos((36 / 180) * Math.PI)
    } L${-r * Math.cos((18 / 180) * Math.PI)} ${
        r * Math.sin((18 / 180) * Math.PI)
    } L${-R * Math.sin((72 / 180) * Math.PI)} ${
        -R * Math.sin(18 / 180) * Math.PI
    } L${-h} ${-r * Math.cos((36 / 180) * Math.PI)} z`;
};
/**
 * 以五角星中心为基准的左半五角星，直接返回path字符串
 * @param R 外切圆半径
 */
export const leftHalfStarPath = (R: number) => {
    const h =
        R /
        (1 / Math.tan((36 / 180) * Math.PI) +
            1 / Math.tan((18 / 180) * Math.PI));
    const r = h / Math.sin((36 / 180) * Math.PI); //内部正五边形的外接圆半径
    return `M0 ${-R} L${-h} ${-r * Math.cos((36 / 180) * Math.PI)} H${
        -R * Math.sin((72 / 180) * Math.PI)
    } L${-r * Math.cos((18 / 180) * Math.PI)} ${
        r * Math.sin((18 / 180) * Math.PI)
    } L${-R * Math.sin((36 / 180) * Math.PI)} ${
        R * Math.cos((36 / 180) * Math.PI)
    } L${0} ${r} L${0} ${-R} z`;
};
/**
 * 以五角星中心为基准的右半五角星，直接返回path字符串
 * @param R 外接圆半径
 */
export const rightHalfStarPath = (R: number) => {
    const h =
        R /
        (1 / Math.tan((36 / 180) * Math.PI) +
            1 / Math.tan((18 / 180) * Math.PI));
    const r = h / Math.sin((36 / 180) * Math.PI); //内部正五边形的外接圆半径
    return `M0 ${-R} L${h} ${-r * Math.cos((36 / 180) * Math.PI)} H${
        R * Math.sin((72 / 180) * Math.PI)
    } L${r * Math.cos((18 / 180) * Math.PI)} ${
        r * Math.sin((18 / 180) * Math.PI)
    } L${R * Math.sin((36 / 180) * Math.PI)} ${
        R * Math.cos((36 / 180) * Math.PI)
    } L${0} ${r} L${0} ${-R} z`;
};

/**
 * 以中心为基准的X形，直接返回path字符串
 * @param R 外切圆半径
 */
export const eksPath = (R: number) => {
    const a = (R / 3) * 2;
    return `M0 ${-a / Math.sqrt(2)} L${a / Math.sqrt(2)} ${
        -a * Math.sqrt(2)
    } l${a / Math.sqrt(2)} ${a / Math.sqrt(2)} L${a / Math.sqrt(2)} 0 l${
        a / Math.sqrt(2)
    } ${a / Math.sqrt(2)} L${a / Math.sqrt(2)} ${a * Math.sqrt(2)} L0 ${
        a / Math.sqrt(2)
    } L${-a / Math.sqrt(2)} ${a * Math.sqrt(2)} l${-a / Math.sqrt(2)} ${
        -a / Math.sqrt(2)
    } L${-a / Math.sqrt(2)} 0 l${-a / Math.sqrt(2)} ${-a / Math.sqrt(2)} L${
        -a / Math.sqrt(2)
    } ${-a * Math.sqrt(2)} L0 ${-a / Math.sqrt(2)} z`;
};
/**
 * 以X形中心为基准的左半X形，直接返回path字符串
 * @param R 外切圆半径
 */
export const leftHalfEksPath = (R: number) => {
    const a = (R / 3) * 2;
    return `M0 ${-a / Math.sqrt(2)} L${-a / Math.sqrt(2)} ${
        -a * Math.sqrt(2)
    } l${-a / Math.sqrt(2)} ${a / Math.sqrt(2)} L${-a / Math.sqrt(2)} 0 l${
        -a / Math.sqrt(2)
    } ${a / Math.sqrt(2)} L${-a / Math.sqrt(2)} ${a * Math.sqrt(2)} L0 ${
        a / Math.sqrt(2)
    } L0 ${-a / Math.sqrt(2)} z`;
};
/**
 * 以X形中心为基准的右半X形，直接返回path字符串
 * @param R 外切圆半径
 */
export const rightHalfEksPath = (R: number) => {
    const a = (R / 3) * 2;
    return `M0 ${-a / Math.sqrt(2)} L${a / Math.sqrt(2)} ${
        -a * Math.sqrt(2)
    } l${a / Math.sqrt(2)} ${a / Math.sqrt(2)} L${a / Math.sqrt(2)} 0 l${
        a / Math.sqrt(2)
    } ${a / Math.sqrt(2)} L${a / Math.sqrt(2)} ${a * Math.sqrt(2)} L0 ${
        a / Math.sqrt(2)
    } L0 ${-a / Math.sqrt(2)} z`;
};

/**
 * 以中心为基准，fill时左半边填充的一个圆，stroke才有轮廓。直接返回path字符串
 * @param R 外接圆半径
 */
export const leftHalfAndWholeCirclePathStroke = (R: number) =>
    `M${0} ${R} A${R} ${R} 0 1 0 ${0} ${-R} v${
        2 * R
    } z A${R} ${R} 0 1 1 ${0} ${-R} a${R} ${R} 0 0 1 ${0} ${2 * R} z`;

/**
 * 以中心为基准，fill时右半边填充的一个圆，stroke才有轮廓。直接返回path字符串
 * @param R 半径
 */
export const rightHalfAndWholeCirclePathStroke = (R: number) =>
    `M${0} ${R} A${R} ${R} 0 1 1 ${0} ${-R} v${
        2 * R
    } z A${R} ${R} 0 1 0 ${0} ${-R} a${R} ${R} 0 0 0 ${0} ${2 * R} z`;

/**
 * 以中心为基准，fill时左半边填充的一个圆，仅fill即可有轮廓。直接返回path字符串
 * @param R 大半径
 * @param border
 */
export const leftHalfAndWholeCirclePathFill = (R: number, border = 1) =>
    `M${0} ${R - border} A${R - border} ${R - border} 0 1 0 ${0} ${-(
        R - border
    )} v${
        2 * R - border
    } z m0 ${border} A${R} ${R} 0 1 1 ${0} ${-R} a${R} ${R} 0 0 1 ${0} ${
        2 * R
    } z`;

/**
 * 以中心为基准，fill时右半边填充的一个圆，仅fill即可有轮廓。直接返回path字符串
 * @param R 大半径
 * @param border
 */
export const rightHalfAndWholeCirclePathFill = (R: number, border = 1) =>
    `M${0} ${R - border} A${R - border} ${R - border} 0 1 1 ${0} ${-(
        R - border
    )} v${
        2 * R - border
    } z m0 ${border} A${R} ${R} 0 1 0 ${0} ${-R} a${R} ${R} 0 0 0 ${0} ${
        2 * R
    } z`;

/**
 * d3.symbol提供了各种图形，但是其size是用面积决定的。\
 * 在scatter中，结点的半径（或尺寸）是需要的。\
 * 因此需要一个函数将length(radius)转换为size(area)\
 * d3推荐的方法是getBBox，需要获取dom\
 * 此函数从几何学进行计算
 * @param symbolName
 * @param R 通常为外接圆半径
 */
export const getAreaBySymbolOuterRadius = (
    symbolName: Type_SymbolName,
    R: number
): number => {
    if (symbolName === "circle") {
        return Math.PI * R * R;
    } else if (symbolName === "cross") {
        return ((R * 2) / 3) * ((R * 2) / 3) * 5; //NOTE 此为外切圆
    } else if (symbolName === "diamond") {
        return (2 / Math.sqrt(3)) * R * R;
    } else if (symbolName === "square") {
        return 2 * R * R;
    } else if (symbolName === "triangle") {
        return ((3 * Math.sqrt(3)) / 4) * R * R;
    } else if (symbolName === "star") {
        return 0.8908130915292852281 * R * R;
        // 5 / (1 / Math.tan(getRadians(36)) + 1 / Math.tan(getRadians(18))))
    } else if (symbolName === "wye") {
        // const k = 1 / Math.sqrt(12);
        // const a = (k / 2 + 1) * 3;
        const r = R / (1 + 1 / Math.sqrt(12));
        return (Math.sqrt(3) / 4) * r * r + 3 * r * r; //NOTE 此为外切圆
    } else {
        return 0;
    }
};
