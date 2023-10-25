<template>
    <div>
        <div class="text">
            {{ props.text }}
        </div>
        <!-- <div class="loader"></div> -->
        <div class="loader2"></div>
    </div>
</template>
<!-- 这个外层的div不建议省略，组件呈现片段或文本根节点，即组件暴露在了最外层，会出现warning: Extraneous non-props attributes -->

<script setup lang="ts">
const props = defineProps({
    text: {
        type: String,
        required: false,
        default: "Loading", //这样在defineAsyncComponent时，如果不传props，则是Loading，
        //若计算过程使用这个组件，则可以传入Running, Running A, Loading B等等
    },
});
</script>

<style scoped>
.text {
    margin: 5% auto;
    text-align: center;
}

.loader,
.loader:before,
.loader:after {
    background: #000000;
    -webkit-animation: load1 1s infinite ease-in-out;
    animation: load1 1s infinite ease-in-out;
    width: 1em;
    height: 4em;
}
.loader {
    color: #000000;
    text-indent: -9999em;
    margin: 88px auto;
    position: relative;
    font-size: 11px;
    -webkit-transform: translateZ(0);
    -ms-transform: translateZ(0);
    transform: translateZ(0);
    -webkit-animation-delay: -0.16s;
    animation-delay: -0.16s;
}
.loader:before,
.loader:after {
    position: absolute;
    top: 0;
    content: "";
}
.loader:before {
    left: -1.5em;
    -webkit-animation-delay: -0.32s;
    animation-delay: -0.32s;
}
.loader:after {
    left: 1.5em;
}
@-webkit-keyframes load1 {
    0%,
    80%,
    100% {
        box-shadow: 0 0;
        height: 4em;
    }
    40% {
        box-shadow: 0 -2em;
        height: 5em;
    }
}
@keyframes load1 {
    0%,
    80%,
    100% {
        box-shadow: 0 0;
        height: 4em;
    }
    40% {
        box-shadow: 0 -2em;
        height: 5em;
    }
}

/************************/
.loader2 {
    margin: 88px auto;
    width: 3em;
    height: 3em;
    border: 5px solid #409eff;
    border-bottom-color: transparent;
    border-radius: 50%;
    display: block;
    box-sizing: border-box;
    animation: rotation 1s linear infinite;
}

@keyframes rotation {
    0% {
        transform: rotate(0deg);
    }
    100% {
        transform: rotate(360deg);
    }
}
</style>
