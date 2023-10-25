# GNNVis

A [Vue](https://vuejs.org/) based visual analytical tool for debugging and refinement on GNN training results, supporting tasks: node classification, link prediction, graph classification, et al.

# Paper

Coming soon...

# Key Features

- Support one model result exploration or comparative analysis between two model results.
- A **dataset** is an output dataset or in other words, a model result, for a complete GNN training task, which will be analyzed by a visual **Dashboard** .
- Each dashboard get data from store and render it in multi sub **View**s from different perspectives: topological space, feature space, latent space, and prediction space.
- Nodes can be selected or highlighted correspondingly between sub **View**s.
- **Nodes Selections** can be **FILTERED** to a new dashboard for further iterative analysis or comparative analysis.
- The process of dashboards generation, which is also the path of analysis, will be stored and visualized by a tree, and each of them can be resumed. **Dashboard-wise comparison** is also supported by select two dashboards in the tree.

# Quick Start

Minimum Requirements:

- CPU Intel® Pentium G4560, Memory 8GB, Storage HDD 10GB or more
- You need [nodejs](https://nodejs.org/en/download/) (version>=16) and [python](https://python.org) (version>=3.7.10).

Clone or download source code and then:

first, run python to start a backend server:

```bash
python ./backend/static_server.py
```

second, start a new bash session and install dependencies, note that you only need to run installation only once:

```bash
npm install
```

third, start the front-end:

```bash
npm run dev
```

Some demo datasets can be found in [/backend/backend.zip](./backend/backend.zip), which should be unzipped in directory `/backend` (choose `yes` if `whether to replace` encountered).

# Online Demo

Coming soon...

# Dataset Format

Please read [this](./docs/data_format.md).

# User Manual

You can read [this](./docs/user_manual/README.md).

If you want to read user manual locally, please follow steps below, since we used [docsify](https://docsify.js.org/) to write user manual.

- If you have [nodejs](https://nodejs.org/en/download/):

    ```bash
    npm i docsify-cli -g
    ```

    then go to project dir, and run:

    ```bash
    docsify serve ./docs/user_manual
    ```

    then open web browser (usually) at [http://localhost:3000](http://localhost:3000) to read the user manual.

- Otherwise, you can

    ```bash
    cd ./docs/user_manual
    python -m http.server 3000
    ```

    then open web browser at [http://localhost:3000](http://localhost:3000) to read the user manual.

# Implementation

## Implementation Overview

- Use [Pinia](https://pinia.vuejs.org/) to provide datasets and global states.
- Use [Vue Router](https://router.vuejs.org/) to manage route.
- Use [Element Plus](https://doc-archive.element-plus.org/#/zh-CN/component/installation) to develop layout components.
- All Vue components are written in [Typescript](https://www.typescriptlang.org/) with [SFC](https://vuejs.org/guide/scaling-up/sfc.html) style and [Composition API](https://vuejs.org/guide/extras/composition-api-faq.html).
- Architecture:
    ![architecture_img](./docs/architecture.png)

## Work Flow

- A model result is placed in `backend` as dirs.
- When entering a route path, `fetch` the model result and store it in `store`'s `DatasetList`.
- Then do some first run calculations and enter into single or comparative `Dashboard`, which is also defined in `store`'s `DashboardList`
- When initializing a new `Dashboard`, we also initialize `View` list in the `Dashboard` given different tasks(node-classification, link-prediction, graph-classification). The `View` list is stored in each `Dashboard` object.
- We put those `View`s in a `Dashboard` as `flex` css layout, with each of the `View` in a `ResizableBox` [slot](https://vuejs.org/guide/components/slots.html#slots) container
- When selecting or highlighting some nodes in different views, we publicly store those infos in `Dashboard` object(not in `View`s) in the `store`.
- When generating a new `Dashboard` with selected nodes, those nodes will be the source nodes of the new `Dashboard`
- `brush`, `zoom`, `resize` and some other behaviors is defined as [Composables](https://vuejs.org/guide/reusability/composables.html) or [Custom Directives](https://vuejs.org/guide/reusability/custom-directives.html#introduction) for code reuse considerations.
- In many scatter plots, we use conditional rendering to render different node coordinates with different semantics.

## Files & Directories Information

```bash
├── LICENSE
├── README.md
├── backend      # the static server and datasets dirs
│   ├── cora-gae # the name of a dataset, also serves as the route path
│   │   ├── graph.json
│   │   ├── initial-layout.json
│   │   ├── node-embeddings-tsne-result.csv
│   │   ├── node-embeddings.csv
│   │   ├── prediction-results.json
│   │   └── true-labels.txt
│   ├── ... # other datasets
│   │   └── ...
│   │
│   ├── gnnvis.py        # data pre-process python file
│   ├── list.json        # the list of datasets, serving as route list in home page
│   └── static_server.py # a simple server
├── docs                 # docs & markdowns
│   ├── architecture.png
│   ├── data_format.md
│   ├── tree.txt
│   └── user_manual.md
├── public
│   ├── favicon.ico           # website icon
│   └── workers               # js files used in web-workers
│       ├── bitset.js         # offline: bitset for Set calculations
│       ├── d3js.org_d3.v7.js # offline: d3 lib
│       ├── tsne.js           # offline: tsne dim reduction algorithm
│       ├── distance.js       # some distance calculation
│       └── forceLayout.js    # the force-directed layout of graphs
├── src
│   ├── App.vue                          # the root vue component
│   ├── api
│   │   └── api.ts                       # the url strings to connect with backend
│   ├── assets                           # root css, not important
│   │   ├── main.css
│   │   └── material-icons.css
│   ├── components                       # vue components
│   │   ├── footer                       # components placed in footer
│   │   │   └── tree.vue                 # history tree view
│   │   ├── header                       # components placed in header
│   │   │   └── scatterSymbolLegend.vue  # the legend of nodes in scatter plots
│   │   ├── icon                         # some icons for buttons
│   │   │   ├── CarbonCenterToFit.vue
│   │   │   └── ...
│   │   ├── plugin                       # plugins for logic reuse
│   │   │   ├── useD3Brush.ts            # d3's brush defined as Vue Composable
│   │   │   ├── useD3Zoom.ts             # d3's zoom  defined as Vue Composable
│   │   │   └── useResize.ts             # resize logics, mainly exposing the box sizes and size changes
│   │   │
│   │   ├── publicViews                     # views for both SingleDashboard and ModelRetsComparativeDashboard
│   │   │   ├── graphHeader.vue             # graph force layout settings in the view box header
│   │   │   ├── graphRenderer.vue           # graph force layout renderer, mainly nodes and links
│   │   │   ├── multiGraphHeader.vue        # multi graph view settings in the view box header
│   │   │   ├── multiGraphRenderer.vue      # multi graph renderer, namely, gridded multi force-directed layout
│   │   │   ├── denseFeatureHeader.vue      # dense feature settings in the view box header
│   │   │   ├── denseFeatureRenderer.vue    # dense feature renderer, mainly multi histograms
│   │   │   ├── sparseFeatureHeader.vue     # node sparse feature settings in the view box header
│   │   │   ├── sparseFeatureRenderer.vue   # node sparse feature renderer, zoomable color matrix
│   │   │   ├── graphFeatureHeader.vue      # graph feature settings in the view box header
│   │   │   ├── graphFeatureRenderer.vue    # graph features, asynchronous, mainly multi histograms
│   │   │   ├── confusionMatrixRenderer.vue # confusion matrix, for both nodes and graphs
│   │   │   ├── linkPredHeader.vue          # link pred settings in the header
│   │   │   ├── linkPredRenderer.vue        # link pred renderer, mainly force-directed layout with colored links
│   │   │   ├── topoLatentDensityHeader.vue # topo-latent density display settings in the view box header
│   │   │   ├── topoLatentDensityRenderer.vue # the topo-latent density renderer, force-directed layout with grey interpolation
│   │   │   ├── tsneHeader.vue              # node emb dim reduction settings in the view box header
│   │   │   ├── tsneRenderer.vue            # node emb dim reduction renderer, namely, scatter
│   │   │   ├── graphTsneHeader.vue         # graph emb dim reduction settings in the view box header
│   │   │   ├── graphTsneRenderer.vue       # graph emb dim reduction renderer, namely, scatter
│   │   │   ├── publicHeader.vue            # public header for view box, with view name and some shared buttons
│   │   │   └── resizableBox.vue            # the resizable view box
│   │   ├── comparison                      # views only for ModelRetsComparativeDashboard
│   │   │   ├── polarHeader.vue             # polar view header
│   │   │   ├── polarRenderer.vue           # polar view renderer, an innovative polar-coord visualization
│   │   │   ├── rankHeader.vue              # rank view header
│   │   │   └── rankRenderer.vue            # an innovative view to visualize the rank diff in 2 emb spaces
│   │   ├── state                           # some public state components
│   │   │   ├── Error.vue
│   │   │   ├── Loading.vue
│   │   │   └── Pending.vue
│   │   ├── singleDashboardView.vue         # dashboard for single model ret
│   │   └── compDashboardView.vue           # dashboard for 2 model rets
│   ├── layouts                             # route layouts
│   │   ├── homeLayout.vue
│   │   ├── notFoundLayout.vue
│   │   ├── compareLayout.vue
│   │   └── singleLayout.vue
│   ├── main.ts                             # web js entrance
│   ├── router
│   │   └── router.ts                       # router definition
│   ├── stores
│   │   ├── enums.ts                        # some enumerative const variables
│   │   └── store.ts                        # global state
│   ├── types
│   │   ├── typeFuncs.ts                    # some type guard functions
│   │   └── types.d.ts                      # type definitions, all in one
│   └── utils                               # helper or pure functions
│       ├── graphUtils.ts                   # some graph calc functions
│       ├── myWebWorker.ts                  # web-worker
│       └── otherUtils.ts                   # some geometric and DOM-related calculation
├── index.html                              # start page
├── package-lock.json                       # node packages full dependency
├── package.json                            # node packages direct dependency
├── env.d.ts                                # typescript settings
├── tsconfig.app.json                       # typescript settings
├── tsconfig.json                           # typescript settings
├── tsconfig.node.json                      # typescript settings
├── vite.config.js                          # building tool settings
└── vite.config.ts                          # building tool settings
```

For more information about files, please refer to the codes.

> We apologize for the mixed Chinese and English docs and comments in the codes. :-)

# FAQ

> trouble shooting

1. What should I do if "Error: 404 File not found"?

- ![404-file-not-fond](./docs/404-file-not-fond.png)
- > This may caused by missing dataset or incomplete dataset, check `backend\list.json` to ensure the dataset is not redundant and check `backend\${your-dataset-name}` to see if the dataset is complete

2. What should I do if "load dataset" process is too long?

- ![load-dataset-too-long](./docs/load-dataset-too-long.png)
- > This may caused by your limited computer performance, try to close irrelevant programs and restart the browser; Or by oversized dataset, for which you can precalculate the graph layout and dimension reduction using other language libs that can be accelerated by GPUs.
