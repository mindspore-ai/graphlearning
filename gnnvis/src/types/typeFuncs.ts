/**
 * @description 类型守卫函数们
 */

import type {
    NodeClassificationPredData,
    GraphClassificationPredData,
    LinkPredictionPredData,
    Dashboard,
    DbWiseComparativeDb,
    CompDashboard,
    SingleDashboard,
} from "./types.d";
export const isTypeNodeClassification = (
    d:
        | NodeClassificationPredData
        | GraphClassificationPredData
        | LinkPredictionPredData
): d is NodeClassificationPredData => d.taskType === "node-classification";
export const isTypeLinkPrediction = (
    d:
        | NodeClassificationPredData
        | GraphClassificationPredData
        | LinkPredictionPredData
): d is LinkPredictionPredData => d.taskType === "link-prediction";
export const isTypeGraphClassification = (
    d:
        | NodeClassificationPredData
        | GraphClassificationPredData
        | LinkPredictionPredData
): d is GraphClassificationPredData => d.taskType === "graph-classification";

export const isDbWiseComparativeDb = (d: Dashboard): d is DbWiseComparativeDb =>
    typeof d.parentId !== "string";
export const isCompDb = (d: Dashboard): d is CompDashboard =>
    Object.hasOwn(d, "refDatasetsNames");
export const isSingleDb = (d: Dashboard): d is SingleDashboard =>
    Object.hasOwn(d, "refDatasetName");
