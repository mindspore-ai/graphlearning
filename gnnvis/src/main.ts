import { createApp } from "vue";
import App from "./App.vue";
import "./assets/main.css";

import { createPinia } from "pinia";
import ElementPlus from "element-plus";
import "element-plus/dist/index.css";
import router from "./router/router";

const app = createApp(App);

app.use(createPinia());
app.use(ElementPlus, { size: "small", zIndex: 3000 });
app.use(router);

app.mount("#app");
