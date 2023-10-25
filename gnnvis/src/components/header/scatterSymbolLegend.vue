<template>
    <div>Scatter Legends</div>
    <svg id="legend" :width="props.width" :height="legendSvgHeight">
        <g
            :transform="`translate(${props.width * margins.left} ${
                legendSvgHeight * margins.top
            })`"
        >
            <g
                v-for="(symbol, i) in legendSymbols"
                :key="i"
                :transform="`translate(${
                    i * (legendRectHeight + legendHorizontalMargin)
                } ${0})`"
            >
                <g
                    :transform="`translate(${legendRectHeight / 2} ${
                        legendRectHeight / 2
                    })`"
                >
                    <path
                        stroke="black"
                        :d=" symbol(legendRectHeight/2)!"
                        fill="black"
                    ></path>
                </g>
                <text
                    :y="legendRectHeight + legendTextMargin"
                    :x="legendRectHeight / 2"
                    text-anchor="middle"
                    :font-size="Math.min(10, legendTextHeight * 0.6)"
                    :dy="0.3 * legendTextHeight"
                >
                    {{ legendSemantics[i] }}
                </text>
            </g>
        </g>
        <g
            :transform="`translate(${props.width * margins.left} ${
                legendSvgHeight * margins.top +
                legendRectHeight +
                legendTextMargin +
                legendTextHeight
            })`"
        >
            <text :font-size="Math.min(12, titleHeight * 0.7)">
                in dashboard-wise comparison:
            </text>
        </g>
        <g
            :transform="`translate(${props.width * margins.left} ${
                legendSvgHeight * margins.top +
                legendRectHeight +
                legendTextMargin +
                legendTextHeight +
                Math.min(12, titleHeight * 0.8)
            })`"
        >
            <g
                v-for="(desc, i) in [
                    'from db0',
                    'from db1',
                    'from db0 and db1 in percents',
                ]"
                :key="i"
                :transform="`translate(${
                    i *
                    (legendRectHeight +
                        (props.width * (1 - margins.left - margins.right) -
                            3 * legendRectHeight) /
                            3)
                } ${0})`"
            >
                <g
                    :transform="`translate(${legendRectHeight / 2} ${
                        legendRectHeight / 2
                    })`"
                >
                    <path
                        stroke="black"
                        :d="circlePath(legendRectHeight / 2)"
                        fill="none"
                    ></path>
                    <path
                        v-if="i === 0"
                        stroke="black"
                        fill="black"
                        :d="leftHalfCirclePath(legendRectHeight / 2)"
                    ></path>
                    <path
                        v-if="i === 1"
                        stroke="black"
                        fill="black"
                        :d="rightHalfCirclePath(legendRectHeight / 2)"
                    ></path>
                    <path
                        v-if="i === 2"
                        stroke="black"
                        fill="black"
                        :d="
                            leftPercentCirclePath(
                                legendRectHeight / 2,
                                Math.random() * (0.4 - 0.2) + 0.2
                            )
                        "
                    ></path>
                    <path
                        v-if="i === 2"
                        stroke="black"
                        fill="black"
                        :d="
                            rightPercentCirclePath(
                                legendRectHeight / 2,
                                Math.random() * (0.4 - 0.2) + 0.2
                            )
                        "
                    ></path>
                </g>
                <text
                    :y="legendRectHeight + legendTextMargin"
                    :x="legendRectHeight / 2"
                    text-anchor="middle"
                    :font-size="Math.min(10, legendTextHeight * 0.6)"
                    :dy="0.3 * legendTextHeight"
                >
                    {{ desc }}
                </text>
            </g>
        </g>
    </svg>
</template>

<script setup lang="ts">
import { computed } from "vue";
import {
    circlePath,
    crossPath,
    diamondPath,
    rectPath,
    trianglePath,
    wyePath,
    starPath,
    eksPath,
    leftHalfCirclePath,
    rightHalfCirclePath,
    leftPercentCirclePath,
    rightPercentCirclePath,
} from "@/utils/otherUtils";
const props = defineProps({
    width: {
        type: Number,
        default: 400,
    },
    height: {
        type: Number,
        default: 300,
    },
});

const legendSvgHeight = computed(() => props.height);
const legendHorizontalMargin = computed(() => 0.02 * props.width);
const margins = { top: 0.03, bottom: 0.03, left: 0.03, right: 0.03 };
const legendRectHeight = computed(() =>
    Math.min(
        0.6 * legendSvgHeight.value,
        (props.width * (1 - margins.left - margins.right) -
            (1 + legendSymbols.length) * legendHorizontalMargin.value) /
            legendSymbols.length
    )
);
const legendTextMargin = computed(() => 0.03 * legendSvgHeight.value);
const legendTextHeight = computed(() => 0.2 * legendSvgHeight.value);
const titleHeight = computed(() => 0.2 * legendSvgHeight.value);
const legendSymbols = [
    circlePath,
    crossPath,
    trianglePath,
    rectPath,
    diamondPath,
    wyePath,
    starPath,
    eksPath,
];
const legendSemantics = [
    "selection",
    "unseen",
    `hop=${1}`,
    `hop=${2}`,
    `hop=${3}`,
    `hop=${4}`,
    `hop=${5}`,
    `hop>${5}`,
];
</script>

<style scoped></style>
