# Documents for Data Descriptions

# Dataset Name

1. 类型：文件夹
2. 简述：该文件夹用于存放一个数据集的多个文件，文件夹的名称即数据集的名称，也可当作 ID，必须唯一，因为它将作为路由路径的一部分。
3. 生成方式：由 python 生成。当 jupyter 环境生成整个工程文件夹（通常是一个 vue-project 的前端工程文件夹）时，在根路径的 public 文件夹下存放这些数据集文件夹。
4. 命名方式：数据集原本的名称-训练模型名称-其他信息，小写英文字母。
    1. 例如：cora-gcn-epoch50、mutag-mpnn-20230331
5. 每个数据集应包含以下文件：
    1. [graph.json](#graph-json)（必须）
    2. [node-embeddings.csv](#node-embeddings-csv)（必须）
    3. [prediction-results.json](#prediction-results-json)（必须）
    4. [graph-embeddings.csv](#graph-embeddings-csv)（多图任务才有）
    5. [initial-layout.json](#initial-layout-json)（必须）
    6. [supplement.txt](#supplement-txt)（如果有）
    7. [graph-custom-index.json](#graph-custom-index-json)(用户自定义的特征，非必须)
    8. [node-dense-features.csv](#node-dense-features-csv)（结点的稠密特征，非必须）
    9. [node-sparse-features.json](#node-sparse-features-json)（结点的稀疏特征，非必须）
    10. [true-labels.txt](#true-labels-txt)（仅当连接预测和图分类时，结点的真实标签，非必须）

## 1. graph.json <span id="graph-json" />

1.类型：文件，参考了 NetworkX（python 包） 2.描述：关于图的原始数据。

```json
{
    "directed": true/false, //是否有向图
    "multigraph": true/false, //是否多图
    "graphs": {
        "0": { //每个图的id，为了方便从0开始逐个递增1即可，亦称索引。单图是否保留id：可以不保留
            "id":"0",
            "label":"",//可选
            "nodes": [ // Array
                0,1,5,17,29,33, //...
                // 该图的节点在所有节点（全集）的中索引/id，从0开始
            ],
            "edges":[ // Array
                0, 9, 50, 1190, //...
                // 该图的边在所有边（全集）中的索引/id，从0开始
                // 无向图，两条边都算
            ]
        },
        "1": {
            //...
        },
    },
    "nodes": [ // Array
        {
            "id": 0,
            "label": 8, //标签
        },
        //...
    ],
    "edges": [
        {
            "source": 0, //出发结点的id
            "target": 1, //目的结点的id
            "eid": 0, //边的id
            "label": 0, //边的标签（可能没有）
        },
        //...
    ],
    "edgeDict": [ //2d-Array 长度等于nodes的长度，含义是对于每个node(按索引），直接相连的边和一阶邻接点。
        [ //第0个node的邻接边、点们。非必须，如不提供则前端自行计算。
            { "nid": 1, "eid": 0 }, //1号node通过0号边和node0相连
            { "nid": 3, "eid": 1 }, //3号node通过1号边和node0相连
            { "nid": 7, "eid": 2 }, //7号node通过2号边和node0相连
            //...
        ],
        [ //... ],//第1个node的邻接边、点们
        //...
    ],
}
```

## 2.node-embeddings.csv <span id="node-embeddings-csv" />

1. 类型：文件
2. 描述：结点在嵌入空间的训练数据，由 MLP 的最后一层生成。每一行是一个结点的 embedding 数据，结点的顺序和 graph.json 中 nodes 的顺序对应，一般按照 id 递增的顺序即可。对于每一行中的数据各维度用英文逗号分隔（可有空格，最后一维不加逗号）。不包含行号，结点 id 等任何其他数据。一个 5 维空间的示例：

```csv
0.130,0.307,0.044,0.032,0.166
0.208,0.510,0.062,0.086,0.112
0.171,0.296,0.262,0.097,0.161
0.104,0.357,0.151,-0.125,0.278
0.374,0.109,-0.143,-0.171,-0.080
//...
```

## 3.prediction-results.json <span id="prediction-results-json" />

1. 类型：文件
2. 描述：模型预测的输出，对于三种不同的任务有不同的格式。

```json
{
    "taskType": "node-classification" | "link-prediction" | "graph-classification",
    //"isLinkPrediction": true,//是否为链接预测任务
    "trueAllowEdges": [//2d-Array 仅当为连接预测任务时才有效
        [0 ,1], // 含义是原图中存在，且预测正确的边，用一个点对表示，即[边的出发点id, 边的终点id]
        [0, 2],
        //...
    ],
    "falseAllowEdges": [//2d-Array 仅当为连接预测任务时才有效
        [0,18],//含义是原图中存在，且预测错误的边，用一个点对表示，即[边的出发点id, 边的终点id]。注意这是原图中的边(ground truth)，而不是预测的结果
        [1,39],
        //...
    ],
    "trueUnseenTopK": 5, //Number, 仅当为连接预测任务时才有效，原图中没有的但推荐出来的边由一个分数排序决定，这个数字表示取前几名
    "trueUnseenEdgesSorted": {//Dict, 仅当为连接预测任务时才有效，原图中没有的但推荐出来的边，
        "11":[44, 161, 3, 19, 339],//每字段一个表示一个结点id，其值为Array，长度为"trueUnseenTopK"，表示给这个节点推荐的边，按分数由高到低排序
        "16": [145, 122, 259, 321, 324],
        //...
    }

    //"isNodeClassification": true,// 是否为结点分类任务
    "numNodeClasses": 7,//Number，仅当结点分类任务时有效，表示结点的种类总共有多少
    "predLabels": [//Array，仅当结点分类任务时有效，表示预测的结点标签，顺序按结点索引（id）从0开始递增的顺序
    ],
    " ": [//Array，仅当结点分类任务时有效，表示真实的结点标签，顺序按结点索引（id）从0开始递增的顺序
    ],

    //"isGraphClassification": true, //是否为图分类任务
    "numGraphClasses": 7, //Number，仅当图任务时有效，表示图的种类总共有多少
    "graphIndex": [//Array，仅当图任务时有效，表示各图的索引(id)，和graph.json中"graph"字段的key一致。非必须，若无，则按0-n顺序递增
    ],
    "predLabels": [//Array，仅当图任务时有效，表示预测的图标签，顺序和graph.json中的"graph"字段的key一致，一般按图索引（id）从0开始递增的顺序
    ],
    "trueLabels": [//Array，仅当图任务时有效，表示真实的图标签，顺序和graph.json中的"graph"字段的key一致，一般按图索引（id）从0开始递增的顺序
    ],
    "phaseDict": {//Dict，仅当图任务时有效，用来映射训练阶段的字典，主要是为了节省空间。非必须。
        0:"train",
        1:"valid",
        2:"predict",
    }
    "phase": [//Array，仅当图任务时有效，每个图参与的训练阶段，顺序和graph.json中的"graph"字段的key一致，一般按图索引（id）从0开始递增的顺序。为节省空间，这里的每个数用"phaseDict"的key
        1,2,0,0,0,1,1,0,//...
    ]
}
```

## 4.graph-embeddings.csv <span id="graph-embeddings-csv" />

1. 类型：文件
2. 描述：仅当图分类任务有效，表示各图在嵌入空间的训练数据，由 MLP 的最后一层生成。每一行是一个图的 embedding 数据，结点的顺序和图索引顺序对应，一般按照 id 递增的顺序即可。对于每一行中的数据各维度用英文逗号分隔（可有空格，最后一维不加逗号）。不包含行号，结点 id 等任何其他数据。

```csv
0.244,-0.146,0.252,-0.240,-0.302
-0.221,-0.431,0.214,0.153,0.007
0.147,-0.199,0.385,0.066,-0.250
0.110,0.171,-0.066,0.849,0.377
//...
```

## 5.initial-layout.json <span id="initial-layout-json" />

1.类型：文件 2.描述：一些计算繁琐的，初始的渲染数据。包括图的力导向图（Force Directed Layout）。嵌入空间数据的降维结果等

```json
{
    "forceDirectedLayout": [
        //Array，长度等于nodes长度，存放图初始渲染时点的坐标
        { "id": 0, "x": 23.1, "y": 50.3 }, //和每个node的id对应，一般为索引
        { "id": 1, "x": 44.5, "y": 90.1 }
        //...
    ],
    "nodeEmbUmp": [
        //2d-Array，长度等于nodes长度，存放结点嵌入空间数据经过Umap降维（二维）后的数据，顺序和结点id对应
        [1.334, 5.132]
        //...
    ],
    "graphEmbUmp": [
        //2d-Array，仅当图分类任务有效(其他任务将此删除），长度等于graph数量，存放图在嵌入空间的数据经过Umap降维（二维）后的数据，顺序和图id对应
        [1.334, 5.132]
        //...
    ]
}
```

3.计算方法介绍：

1. 对于 forceDirectedLayout，可参考<https://github.com/d3/d3-force/tree/v3.0.0#d3-force>
2. 对于 Umap, Tsne，可使用 python 的 sklearn

## 6.supplement.txt <span id="supplement-txt" />

1. 类型：文件
2. 描述：关于此数据集的其他任何信息，例如本次训练的超参数。由人工撰写，非程序生成。

## 7.graph-custom-index.json <span id="graph-custom-index-json" />

1. 类型：文件
2. 描述：用户自定义的，通过后端临时计算的特征数据。

```json
{
    "index_target": "graph"|"node",//用什么做索引，graph的id或者node的id
    "number_of_C": {  //某个特征（feature）或指标（metric）的名称，对应一个dict
        "0": 14,   //dict中每一个键值对是graph或node的id（string类型），值为特征值（number）。数量和graph或node对应。
        "1": 9,
        "2": 9,
        "3": 16,
        "4": 6,
        ...
     },
     "number_of_F": {
         ...
     },
     ...
}
```

## 8.node-dense-features.csv <span id="node-dense-features-csv" />

1. 类型：文件
2. 描述：结点 的稠密特征。每个结点可能多个特征，但是每个特征都是一个数字(scala)。第一行是特征的名称（非必须）。后面 2-n 行和 node 数量对应，值为结点特征

```csv
"np.random.normal(5,2,2708)","np.random.poisson(1,2708)","np.random.triangular(-1,15,20,2708)" //字符串，每个字符串代表一个特征的名称，用逗号分隔
1.34,2.0,18.628 //0号node的特征，和特征名称一一对应，也用逗号分隔
1.80,0.0,18.96
4.20,0.0,5.488
3.99,1.0,18.884
...
```

## 9.node-sparse-features.json <span id="node-sparse-features-json" />

1.类型：文件 2.描述：结点的稀疏特征。为了节省空间使用 json，仅存放非空数据。

```json
{
    "numNodeFeatureDims":888,//number，特征的维度数
    "nodeFeatureIndexes": [//2d-array 索引顺序和结点id对应，每个subArray是Array<number>，表示当前结点在众多特征维度上哪些索引（从零开始）上值是非空的。
        [65,77,391,801], //0号结点在特征的第65维、第77维、第391维、第801维有数据
        [30,102,887],//1号结点在特征的30维、第102维、第887维有数据
        ...
    ],
    "nodeFeatureValues":[    //2d-array，索引顺序和结点id对应，每个subArray是Array<number>，和nodeFeatureIndexes对应，表示当前结点在这些维度上的特征值是多少。
        [1.2, 1.5 , 1.8, 0.3], //0号结点在特征的第65维上值是1.2、在第77维上的值是1.5、第391维上的值是1.8、第801维上的值是0.3
        [0.2, 1.4, 1.5],//1号结点在特征的30维上值是0.2、第413维上的值是1.4、第887维上的值是1.5
        ...
    ]
}
```

## 10. true-labels.txt <span id="true-labels-txt" />

1. 类型：文件
2. 描述： 仅当连接预测和图分类时，结点的真实标签，非必须。每一行和结点或图的 id 对应

```txt
7
0
3
2
...
```

# Python Object 格式

调用如下函数以启动可视化窗口：

```python
window = GNNVis(
    graph_type,
    graphs,
    node_embed = node_embed,
    link_pred_res = link_pred_res,
    node_classify_res = node_classify_res,
    graph_classify_res = graph_classify_res,
    graph_embed = graph_embed,
    gen_path = gen_path
)
```

各参数按照功能分类，格式描述如下：

## 1. 图的表示

### graph_type

一个整数，取值范围 `[0, 1, 2, 3]`，表示两个二进制位。

| 数值 | 高位（是否有重边） | 低位（是否是有向图） |
| ---- | ------------------ | -------------------- |
| 0    | 无重边             | 无向图               |
| 1    | 有重边             | 有向图               |

### graphs

如果源数据支持直接使用 networkx 格式，最好直接使用；否则按照如下格式定义：

```python
graphs = [
    ...,
    {  # graph[i] 表示编号为 i 的图所含点、边的信息
        "id": int,  # 第 i 张图的编号
        "g_value": Any,  # 第 i 张图的全局信息；若没有，可无此键值对
        "nodes" : [
            # 图包含的所有点的信息
            # 若无点信息，每个元素为一个 int，表示点的 id
            # 若有点信息，每个元素为 list[int, dict]，表示 [点编号, 点属性]
        ],
        "links": [
            # 图包含的所有边的 id
            # 每个元素为一个 list[int, int, dict]，表示 [起点, 终点, 边属性]
            # 边属性 dict 中必须包含字段 eid 表示边编号
        ]
    },
    ...
]
```

## 2.模型预测输出

### link_pred_res

类型：dict，默认为 None，仅当有连接预测结果时传入；
字典键值描述如下：

```python
link_pred_res = {
    "trueAllowEdges": [
        # 元素类型：list[int, int]
        # 原图中存在，且预测正确的边，用一个点对表示，即 [边的出发点id, 边的终点id]
    ],
    "falseAllowEdges": [
        # 元素类型：list[int, int]
        # 原图中存在，且预测错误的边，用一个点对表示，即 [边的出发点id, 边的终点id]
        # 注意:这是原图中的边(ground truth)，而不是预测结果
    ],
    "trueUnseenTopK": 5,  # int，原图中没有的但推荐出来的边由一个分数排序决定，这个数字表示取前几名
    "trueUnseenEdgesSorted": {
        # 原图中没有的但推荐出来的边
        # key: list[int]
        # 每字段一个表示一个结点 id，其值为 Array，长度为 "trueUnseenTopK"，表示给这个节点推荐的边，按分数由高到低排序
    }
}
```

### node_classify_res

类型：dict，默认为 None，仅当有节点分类结果时传入。
字典键值描述如下：

```python
node_classify_res = {
    "numNodeClasses": int,  # 结点的种类总数
    "predLabels": [
        # int
        # 预测的结点标签，顺序按结点索引 id 从 0 开始递增的顺序
    ],
    "trueLabels": [  # list，真实的结点标签，顺序按结点索引 id 从 0 开始递增的顺序
    ],
}
```

### graph_classify_res

类型：dict，默认为 None，仅当有图分类结果时传入。
字典键值描述如下：

```python
graph_classify_res = {
    "numGraphClasses": int,  # int，图的种类总数
    "graph_index": [  # list，各图的索引 id ，和 graph.json 中 "graph" 字段的 key 一致
    ],
    "predLabels": [  # list，预测的图标签，顺序和 graph.json 中的 "graph" 字段的 key 一致，一般按图索引 id 从 0 开始递增的顺序
    ],
    "trueLabels": [  # list，真实的图标签，顺序和 graph.json 中的 "graph" 字段的 key 一致，一般按图索引 id 从 0 开始递增的顺序
    ],
    "phaseDict": {  # dict，映射训练阶段的字典，以节省空间
        0: "train",
        1: "valid",
        2: "predict",
    },
    "phase": [
        # list[int]
        # 每个图参与的训练阶段，顺序和 graph.json 中的 "graph" 字段的 key 一致，一般按图索引 id 从 0 开始递增的顺序
        # 每个数用 "phaseDict" 的 key
    ]
}
```

### node_embed

`np.ndarray` 二维数组

### graph_embed

`np.ndarray` 二维数组
