# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Dataset"""
import numpy as np
from mindspore_gl.sampling import sage_sampler_on_homo
from mindspore_gl.dataloader import Dataset
import mindspore_gl.array_kernel as array_kernel
from mindspore_gl.graph.ops import PadArray2d, PadMode, PadDirection


class GraphSAGEDataset(Dataset):
    """GraphSage Dataset"""
    def __init__(self, graph_dataset, neighbor_nums, batch_size):
        self.graph_dataset = graph_dataset
        self.graph = graph_dataset[0]
        self.neighbor_nums = neighbor_nums
        self.x = graph_dataset.node_feat
        self.y = graph_dataset.node_label
        self.batch_size = batch_size

        self.max_sampled_nodes_num = neighbor_nums[0] * neighbor_nums[1] * batch_size

        self.layered_edges_0_pad_op = PadArray2d(mode=PadMode.CONST, size=[2, neighbor_nums[0] * self.batch_size],
                                                 dtype=np.int32, direction=PadDirection.ROW,
                                                 fill_value=self.max_sampled_nodes_num - 1
                                                 )
        self.layered_edges_1_pad_op = PadArray2d(mode=PadMode.CONST, size=[2, self.max_sampled_nodes_num],
                                                 dtype=np.int32, direction=PadDirection.ROW,
                                                 fill_value=self.max_sampled_nodes_num - 1,
                                                 )
        self.nid_feat_pad_op = PadArray2d(mode=PadMode.CONST,
                                          size=[self.max_sampled_nodes_num, graph_dataset.num_features],
                                          dtype=graph_dataset.node_feat.dtype,
                                          direction=PadDirection.COL,
                                          fill_value=0,
                                          reset_with_fill_value=False,
                                          use_shared_numpy=True
                                          )

    def __getitem__(self, batch_nodes):
        res = sage_sampler_on_homo(self.graph, batch_nodes, self.neighbor_nums)
        label = array_kernel.int_1d_array_slicing(self.y, batch_nodes)
        # start_time = time.time()
        ###############################################
        # Pad Edge Array
        ##############################################
        layered_edges_0 = self.layered_edges_0_pad_op(res['layered_edges_0'])
        layered_edges_1 = self.layered_edges_1_pad_op(res['layered_edges_1'])
        ########################
        # Pad Node Feat
        ########################
        feat = self.nid_feat_pad_op.lazy([len(res['all_nodes']), self.graph_dataset.num_features])
        array_kernel.float_2d_gather_with_dst(feat, self.graph_dataset.node_feat, res['all_nodes'])
        # print(f"pad time = {time.time() - start_time}")
        return res['seeds_idx'], label, feat, layered_edges_0, layered_edges_1
