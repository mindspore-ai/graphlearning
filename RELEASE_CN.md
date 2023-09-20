# MindSpore Graph Learning Release Notes

[View English](./RELEASE.md)

## MindSpore Graph Learning 0.2.0 Release Notes

### 主要特性及增强

- [STABLE] 新增了基于CSR稀疏算子的加速方案，涵盖了从CSR数据格式转换、CSR图填充到CSR消息聚合等方面。
- [STABLE] 提供了基于CSR稀疏算子的APPNP、GCN、GAT、GATv2、MPNN的训练示例。

### API变更

#### 新增/增强API

##### Python APIs

- 新增graph接口 `mindspore_gl.graph.batch_graph_csr_data` 。
- 新增graph接口 `mindspore_gl.graph.graph_csr_data` 。
- 新增graph接口 `mindspore_gl.graph.PadCsrEdge` 。
- 新增graph接口 `mindspore_gl.graph.sampling_csr_data` 。

- 新增nn接口 `mindspore_gl.nn.GNNCell.sparse_compute` 。

#### 非兼容性变更

##### Python APIs

- `mindspore_gl.GraphField` 新增参数 `indices` 、 `indptr` 、 `indices_backward` 、 `indptr_backward` 、 `csr` ，设置是否使用稀疏模式。[(!214)](https://gitee.com/mindspore/graphlearning/pulls/214)

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v0.2.0 </td>
  </tr>
  <tr>
  <td><pre>
  n_nodes = 7
  n_edges = 8
  src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
  dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
  graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
  </pre>
  </td>
  <td><pre>
  n_nodes = 7
  n_edges = 8
  indices = ms.Tensor([2, 3, 5, 6, 3, 4, 0, 6], ms.int32)
  indptr = ms.Tensor([0, 2, 4, 5, 6, 7, 8, 8], ms.int32)
  indices_backward = ms.Tensor([4, 0, 0, 2, 3, 1, 1, 5], ms.int32)
  indptr_backward = ms.Tensor([0, 1, 1, 2, 4, 5, 6, 8], ms.int32)
  graph_field = GraphField
    (n_nodes=n_nodes, n_edges=n_edges, indices=indices, indptr=indptr,
  ...
     indices_backward=indices_backward, indptr_backward=indptr_backward, csr=True)
  </pre>
  </td>
  </tr>
  </table>

- `mindspore_gl.BatchedGraphField` 新增参数 `indices` 、 `indptr` 、 `indices_backward` 、 `indptr_backward` 、 `csr` ，设置是否使用稀疏模式。[(!214)](https://gitee.com/mindspore/graphlearning/pulls/214)

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v0.2.0 </td>
  </tr>
  <tr>
  <td><pre>
  n_nodes = 9
  n_edges = 11
  src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6, 8, 8, 8], ms.int32)
  dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4, 8, 8, 8], ms.int32)
  ver_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 2, 2], ms.int32)
  edge_subgraph_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2], ms.int32)
  graph_mask = ms.Tensor([1, 1, 0], ms.int32)
  graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx,
  ...                             edge_subgraph_idx, graph_mask)
  </pre>
  </td>
  <td><pre>
  n_nodes = 9
  n_edges = 11
  indices = ms.Tensor([0, 3, 4, 6, 8, 4, 5, 1, 8], ms.int32)
  indptr = ms.Tensor([0, 1, 3, 5, 6, 7, 8, 9, 9, 9], ms.int32)
  indices_backward = ms.Tensor([0, 5, 1, 1, 3, 4, 2, 2, 6], ms.int32)
  indptr_backward = ms.Tensor([0, 1, 2, 2, 3, 5, 6, 7, 7, 9], ms.int32)
  node_map_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1], ms.int32)
  edge_map_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 1, 1], ms.int32)
  graph_mask = ms.Tensor([1, 0], ms.int32)
  graph_field = BatchedGraphField
    (indices=indices, indptr=indptr, indices_backward=indices_backward,
     indptr_backward=indptr_backward, csr=True, n_nodes=n_nodes,
  ...
  n_edges=n_edges, ver_subgraph_idx=node_map_idx,
  ...
  edge_subgraph_idx=edge_map_idx, graph_mask=graph_mask)
  </pre>
  </td>
  </tr>
  </table>

### 贡献者

感谢以下人员做出的贡献:

James Cheng, yufan, wuyidi, yinpeiqi, liuxiulong, wangqirui, chengbin, luolan, zhengzuohe, lujiale, liyang, huenrui, baocong, zhangqinghua, wangyushan, zhushujing, zhongjicheng, gaoxiang, yushunmin, fengxun, gongyue, wangyixuan, zuochuanyong, yuhan, wangying, chujinjin, xiezuoquan, yeyuhang, xuhn1997.

欢迎以任何形式对项目提供贡献！

## MindSpore Graph Learning 0.2.0-alpha Release Notes

### 主要特性及增强

- [STABLE] 新增30+ GNN API用于图卷积、池化和其他图操作，如填充、归一化和采样等。
- [STABLE] 新增多个数据集API，包含整图（Reddit、BlogCatalog）、批次图（Alchemy、Enzymes、IMDBBinary和PPI）和时空图（MetrLa）。
- [STABLE] 新增使用MindSpore Graph Learning的经典GNN模型的训练示例，包括图游走（deepwalk、geniepath）、生物化学（diffpool、mpnn）、社交网络（gin、graphsage）、图自动编码器（gae、vgae）和时空图（stgcn）。
- [STABLE] 提供了在Ascend和GPU上GNN采样和数据并行训练的分布式示例。

### API变更

#### 新增/增强API

##### Python APIs

- 新增dataloader接口 `mindspore_gl.dataloader.split_data` 。
- 新增dataloader接口 `mindspore_gl.dataloader.RandomBatchSampler` 。
- 新增dataloader接口 `mindspore_gl.dataloader.Dataset` 。

- 新增dataset接口 `mindspore_gl.dataset.Alchemy` 。
- 新增dataset接口 `mindspore_gl.dataset.BlogCatalog` 。
- 新增dataset接口 `mindspore_gl.dataset.Enzymes` 。
- 新增dataset接口 `mindspore_gl.dataset.IMDBBinary` 。
- 新增dataset接口 `mindspore_gl.dataset.MetrLa` 。
- 新增dataset接口 `mindspore_gl.dataset.PPI` 。
- 新增dataset接口 `mindspore_gl.dataset.Reddit` 。

- 新增graph接口 `mindspore_gl.graph.add_self_loop` 。
- 新增graph接口 `mindspore_gl.graph.get_laplacian` 。
- 新增graph接口 `mindspore_gl.graph.norm` 。
- 新增graph接口 `mindspore_gl.graph.remove_self_loop` 。
- 新增graph接口 `mindspore_gl.graph.BatchHomoGraph` 。
- 新增graph接口 `mindspore_gl.graph.BatchMeta` 。
- 新增graph接口 `mindspore_gl.graph.CsrAdj` 。
- 新增graph接口 `mindspore_gl.graph.MindHomoGraph` 。
- 新增graph接口 `mindspore_gl.graph.PadArray2d` 。
- 新增graph接口 `mindspore_gl.graph.PadDirection` 。
- 新增graph接口 `mindspore_gl.graph.PadHomoGraph` 。
- 新增graph接口 `mindspore_gl.graph.PadMode` 。
- 新增graph接口 `mindspore_gl.graph.UnBatchHomoGraph` 。

- 新增nn接口 `mindspore_gl.nn.AGNNConv` 。
- 新增nn接口 `mindspore_gl.nn.ASTGCN` 。
- 新增nn接口 `mindspore_gl.nn.AvgPooling` 。
- 新增nn接口 `mindspore_gl.nn.CFConv` 。
- 新增nn接口 `mindspore_gl.nn.ChebConv` 。
- 新增nn接口 `mindspore_gl.nn.DOTGATConv` 。
- 新增nn接口 `mindspore_gl.nn.EDGEConv` 。
- 新增nn接口 `mindspore_gl.nn.EGConv` 。
- 新增nn接口 `mindspore_gl.nn.GatedGraphConv` 。
- 新增nn接口 `mindspore_gl.nn.GATv2Conv` 。
- 新增nn接口 `mindspore_gl.nn.GCNConv2` 。
- 新增nn接口 `mindspore_gl.nn.GINConv` 。
- 新增nn接口 `mindspore_gl.nn.GlobalAttentionPooling` 。
- 新增nn接口 `mindspore_gl.nn.GMMConv` 。
- 新增nn接口 `mindspore_gl.nn.MaxPooling` 。
- 新增nn接口 `mindspore_gl.nn.MeanConv` 。
- 新增nn接口 `mindspore_gl.nn.NNConv` 。
- 新增nn接口 `mindspore_gl.nn.SAGEConv` 。
- 新增nn接口 `mindspore_gl.nn.SAGPooling` 。
- 新增nn接口 `mindspore_gl.nn.Set2Set` 。
- 新增nn接口 `mindspore_gl.nn.SGConv` 。
- 新增nn接口 `mindspore_gl.nn.SortPooling` 。
- 新增nn接口 `mindspore_gl.nn.STConv` 。
- 新增nn接口 `mindspore_gl.nn.SumPooling` 。
- 新增nn接口 `mindspore_gl.nn.TAGConv` 。
- 新增nn接口 `mindspore_gl.nn.WeightAndSum` 。

- 新增sampling接口 `mindspore_gl.sampling.negative_sample` 。
- 新增sampling接口 `mindspore_gl.sampling.random_walk_unbias_on_homo` 。
- 新增sampling接口 `mindspore_gl.sampling.sage_sampler_on_homo` 。

- 新增utils接口 `mindspore_gl.utils.pca` 。

### 贡献者

感谢以下人员做出的贡献:

James Cheng, yufan, wuyidi, yinpeiqi, liuxiulong, wangqirui, chengbin, luolan, zhengzuohe, lujiale, liyang, huenrui, baocong, zhangqinghua, wangyushan, zhushujing, zhongjicheng, gaoxiang, yushunmin, fengxun, gongyue, wangyixuan, zuochuanyong, yuhan, wangying, chujinjin, xiezuoquan, yeyuhang, xuhn1997.

欢迎以任何形式对项目提供贡献！
