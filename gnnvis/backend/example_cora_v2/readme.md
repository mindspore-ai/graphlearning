# Demo: GAE training code based on CoraV2

The MAIN python script is `/example_cora_v2/model_gae/trainval.py`.

> The **GNNVis** module is in the upper-level directory `backend`

Ensure that you are in the folder `example_cora_v2`, then run the shell script `run_trainval_gae.sh`.

If your **sparse** features are represented in the data stream as a two-dimensional table **(dense, csv file)**, you can use `dense_to_sparse.py` to convert it to **sparse** format **(json file)**.
