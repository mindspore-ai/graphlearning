# Contents

- Accelerate GraphLearning Using Compressed Sparse Row (CSR) Computing
- Datasets
- Environment Requirements
- Quick Start
- Experiment results (ms/step)

## Accelerate GraphLearning Using Compressed Sparse Row (CSR) Computing

a. What is CSR and why is it useful

The CSR format represents a tensor by three 1-D arrays, that respectively contain the nonzero values, the extents of rows, and column indices. It compresses the row indices and allows fast row access. For a sparse tensor with shape (m, n) and number of nonzero values nnz, the value array and the column indices array both have size nnz, and contain non-zero values and column indices of those values respectively. The row index pointer is of length
m + 1 and encodes the index in the value array and the column indices array where the given row starts (Click [here](https://mindspore.cn/docs/en/r1.8/api_python/mindspore/mindspore.CSRTensor.html?highlight=csrtensor) to read more about CSRTensor in mindspore).

Compared to COO format, for which the row indices are not compressed, CSR has the advantage of lower memory consumption and continuous access in each row. And by sorting the rows based on the number of values in each row, load balancing can be achieved by assigning adjacent rows (with similar number of values) to concurrent streams (Click [here](http://dl.acm.org/doi/pdf/10.1145/3447786.3456247) to view the paper on optimizing GNN through vertex-centric programming by Y Wu).

b. Optimizing GNN models using CSR

For large datasets (e.g. reddit with over 100,000 vertices), using the csr version has more than 50% speed-up over the original version, and in some cases can be more than 1.3 times faster.

For small datasets however (e.g. corav2 with less than 3000 vertices), the csr version is not guaranteed to render better performance, since the calculation overhead is relatively large.

c. Generating highly optimized CSR operators via Automatic Kernel Generator (click here to read more about [AKG](https://gitee.com/mindspore/akg))

CSR operators are compiled using AKG, which automatically generates efficient code based on types and shapes of the inputs. CSR operators may fuse with other operators to reduce data transfer and memory consumption. Each row in a CSR tensor is mapped to a single block with multiple warps to make use of coalesced access on the GPU.

## Datasets

The experiment is based on cora_v2_with_mask.npz and reddit_with_mask.npz

## Environment Requirements

- MindSpore >= 1.8.0
- GPU

## Quick Start

bash run_bench.sh

## Experiment results

Performance comparison (ms/step)

| Dataset | corav2 | reddit |
| :----: | :----: | :----: |
| APPNP original | 1.63 | 890 |
| APPNP csr | 2.09 | 374.5 |
| GAT original | 1.63 | 850 |
| GAT csr | 1.43 | 432.18 |
| GCN original | 1.61 | 125 |
| GCN csr | 1.58 | 78.5 |
