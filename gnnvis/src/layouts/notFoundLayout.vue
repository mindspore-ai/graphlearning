<template>
    <!-- <h1>Not Found</h1>
    <RouterLink to="/home">Back to Home</RouterLink> -->
    <el-empty description="Not Found">
        <RouterLink to="/home" :replace="true">
            <el-button type="primary">Back to Home</el-button></RouterLink
        >
    </el-empty>
</template>
<style scoped>
h1 {
    text-align: center;
}
/* a,
a:link,
a:visited,
a:hover,
a:active {
    text-align: center;
    text-decoration: none;
} */
</style>
