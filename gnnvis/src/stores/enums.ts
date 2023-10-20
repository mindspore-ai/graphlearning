import * as d3 from "d3";

/**
 * 任务类型
 */
export const taskTypes = [
    "node-classification",
    "link-prediction",
    "graph-classification",
] as const;
/**
 * 标签类型
 */
export const labelTypes = ["true", "pred"] as const;
/**
 * rankView中所选点或点集计算embed差异时，对所选的点使用何种统计方法
 */
export const rankEmbDiffAlgos = ["single", "average", "center"] as const;
/**
 * 所选点或点集计算embed差异时，对所选的点使用何种统计方法
 */
export const polarEmbDiffAlgos = ["single", "average", "center"] as const;
/**
 * polar坐标系中衡量topo空间差异的算法
 */
export const polarTopoDistAlgos = [
    "shortest path",
    "hamming",
    "jaccard",
] as const;
/**
 * 多个dashboard的布局方式，在下面跟着显示，还是只有一个窗格切换
 */
export const dashboardsLayoutModes = ["append", "replace"] as const;
/**
 * 是否手动清除，即每次select接着选还是重新选
 */
export const clearSelModes = ["manual", "auto"] as const;
/**
 * view缩放时，何时重新计算svg中的坐标，瞬时或resize结束时\
 */
export const whenToRescaleModes = ["simultaneously", "onResizeEnd"] as const;
/**
 * 那些setting齿轮，是悬浮触发还是点击触发
 */
export const settingsMenuTriggerModes = ["hover", "click"] as const;

/**
 * scatter结点的symbol的名称们
 */
export const symbolNames = [
    "circle",
    "triangle",
    "cross",
    "diamond",
    "square",
    "star",
    "triangle",
    "wye",
] as const;
