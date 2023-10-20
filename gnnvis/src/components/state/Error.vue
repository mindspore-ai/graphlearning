<template>
    <div style="color: red">
        <div
            v-if="props['font-size']"
            :style="{ fontSize: props['font-size'] }"
        >
            {{ props.error }}
        </div>
        <h1 v-else>
            {{
                // Object.hasOwn(props.error, "message")
                //     ? props.error.message
                // : props.error
                props.error
            }}
        </h1>
    </div>
</template>

<script setup lang="ts">
const props = defineProps(["error", "font-size"]);
</script>

<style scoped></style>
