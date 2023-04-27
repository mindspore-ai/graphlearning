# MindSpore Graph Learning Release Notes

[查看中文](./RELEASE_CN.md)

## MindSpore Graph Learning 0.2.0 Release Notes

### Major Features and Improvements

- [STABLE] Add a CSR sparse operator-based acceleration solution, covering CSR data format conversion, CSR graph padding, and CSR message aggregation.
- [STABLE] Provide training examples of APPNP, GCN, GAT, GATv2, and MPNN based on the CSR sparse operator.

### API Change

#### New APIs & Enhanced APIs

##### Python APIs

- Add graph API `mindspore_gl.graph.batch_graph_csr_data` 。
- Add graph API `mindspore_gl.graph.graph_csr_data` 。
- Add graph API `mindspore_gl.graph.PadCsrEdge` 。
- Add graph API `mindspore_gl.graph.sampling_csr_data` 。

- Add nn API `mindspore_gl.nn.GNNCell.sparse_compute` 。

#### Incompatible Modification

##### Python APIs

- `mindspore_gl.GraphField` add the parameter `indices` 、 `indptr` 、 `indices_backward` 、 `indptr_backward` 、 `csr`，set to use sparse mode. [(!214)](https://gitee.com/mindspore/graphlearning/pulls/214)

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> v0.2.0 </td>
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
  graph_field = GraphField(n_nodes=n_nodes, n_edges=n_edges, indices=indices, indptr=indptr,
  ...                      indices_backward=indices_backward, indptr_backward=indptr_backward, csr=True)
  </pre>
  </td>
  </tr>
  </table>

- `mindspore_gl.BatchedGraphField` add the parameter `indices` 、 `indptr` 、 `indices_backward` 、 `indptr_backward` 、 `csr`，set to use sparse mode. [(!214)](https://gitee.com/mindspore/graphlearning/pulls/214)

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> v0.2.0 </td>
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
  graph_field = BatchedGraphField(indices=indices, indptr=indptr, indices_backward=indices_backward,
                                  indptr_backward=indptr_backward, csr=True, n_nodes=n_nodes,
  ...                             n_edges=n_edges, ver_subgraph_idx=node_map_idx,
  ...                             edge_subgraph_idx=edge_map_idx, graph_mask=graph_mask)
  </pre>
  </td>
  </tr>
  </table>

### Contributors

Thanks goes to these wonderful people:

James Cheng, yufan, wuyidi, yinpeiqi, liuxiulong, wangqirui, chengbin, luolan, zhengzuohe, lujiale, liyang, huenrui, baocong, zhangqinghua, wangyushan, zhushujing, zhongjicheng, gaoxiang, yushunmin, fengxun, gongyue, wangyixuan, zuochuanyong, yuhan, wangying, chujinjin, xiezuoquan, yeyuhang, xuhn1997.

Contributions of any kind are welcome!

## MindSpore Graph Learning 0.2.0-alpha Release Notes

### Major Features and Improvements

- [STABLE] Add 30+ GNN API for graph conv, pooling operations and other operations, such as padding, normalization, sampling.
- [STABLE] Add dataset API contains whole graph (Reddit, BlogCatalog), batched graph (Alchemy, Enzymes, IMDBBinary and PPI) and spatial-temporal graph (MetrLa).
- [STABLE] Add training examples of typical GNN models using MindSpore Graph Learning including Graph Walking (deepwalk, geniepath), Biochemistry (diffpool, mpnn), Social Network (gin, graphsage), Graph Auto Encoder (gae, vgae) and Spatio-Temporal Graph (stgcn).
- [STABLE] Provide distributed examples for GNN sampling and training with data parallelism in Ascend and GPU.

### API Change

#### New APIs & Enhanced APIs

##### Python APIs

- Add dataloader API `mindspore_gl.dataloader.split_data` .
- Add dataloader API `mindspore_gl.dataloader.RandomBatchSampler` .
- Add dataloader API `mindspore_gl.dataloader.Dataset` .

- Add dataset API `mindspore_gl.dataset.Alchemy` .
- Add dataset API `mindspore_gl.dataset.BlogCatalog` .
- Add dataset API `mindspore_gl.dataset.Enzymes` .
- Add dataset API `mindspore_gl.dataset.IMDBBinary` .
- Add dataset API `mindspore_gl.dataset.MetrLa` .
- Add dataset API `mindspore_gl.dataset.PPI` .
- Add dataset API `mindspore_gl.dataset.Reddit` .

- Add graph API `mindspore_gl.graph.add_self_loop` .
- Add graph API `mindspore_gl.graph.get_laplacian` .
- Add graph API `mindspore_gl.graph.norm` .
- Add graph API `mindspore_gl.graph.remove_self_loop` .
- Add graph API `mindspore_gl.graph.BatchHomoGraph` .
- Add graph API `mindspore_gl.graph.BatchMeta` .
- Add graph API `mindspore_gl.graph.CsrAdj` .
- Add graph API `mindspore_gl.graph.MindHomoGraph` .
- Add graph API `mindspore_gl.graph.PadArray2d` .
- Add graph API `mindspore_gl.graph.PadDirection` .
- Add graph API `mindspore_gl.graph.PadHomoGraph` .
- Add graph API `mindspore_gl.graph.PadMode` .
- Add graph API `mindspore_gl.graph.UnBatchHomoGraph` .

- Add nn API `mindspore_gl.nn.AGNNConv` .
- Add nn API `mindspore_gl.nn.ASTGCN` .
- Add nn API `mindspore_gl.nn.AvgPooling` .
- Add nn API `mindspore_gl.nn.CFConv` .
- Add nn API `mindspore_gl.nn.ChebConv` .
- Add nn API `mindspore_gl.nn.DOTGATConv` .
- Add nn API `mindspore_gl.nn.EDGEConv` .
- Add nn API `mindspore_gl.nn.EGConv` .
- Add nn API `mindspore_gl.nn.GatedGraphConv` .
- Add nn API `mindspore_gl.nn.GATv2Conv` .
- Add nn API `mindspore_gl.nn.GCNConv2` .
- Add nn API `mindspore_gl.nn.GINConv` .
- Add nn API `mindspore_gl.nn.GlobalAttentionPooling` .
- Add nn API `mindspore_gl.nn.GMMConv` .
- Add nn API `mindspore_gl.nn.MaxPooling` .
- Add nn API `mindspore_gl.nn.MeanConv` .
- Add nn API `mindspore_gl.nn.NNConv` .
- Add nn API `mindspore_gl.nn.SAGEConv` .
- Add nn API `mindspore_gl.nn.SAGPooling` .
- Add nn API `mindspore_gl.nn.Set2Set` .
- Add nn API `mindspore_gl.nn.SGConv` .
- Add nn API `mindspore_gl.nn.SortPooling` .
- Add nn API `mindspore_gl.nn.STConv` .
- Add nn API `mindspore_gl.nn.SumPooling` .
- Add nn API `mindspore_gl.nn.TAGConv` .
- Add nn API `mindspore_gl.nn.WeightAndSum` .

- Add sampling API `mindspore_gl.sampling.negative_sample` .
- Add sampling API `mindspore_gl.sampling.random_walk_unbias_on_homo` .
- Add sampling API `mindspore_gl.sampling.sage_sampler_on_homo` .

- Add utils API `mindspore_gl.utils.pca` .

### Contributors

Thanks goes to these wonderful people:

James Cheng, yufan, wuyidi, yinpeiqi, liuxiulong, wangqirui, chengbin, luolan, zhengzuohe, lujiale, liyang, huenrui, baocong, zhangqinghua, wangyushan, zhushujing, zhongjicheng, gaoxiang, yushunmin, fengxun, gongyue, wangyixuan, zuochuanyong, yuhan, wangying, chujinjin, xiezuoquan, yeyuhang, xuhn1997.

Contributions of any kind are welcome!
