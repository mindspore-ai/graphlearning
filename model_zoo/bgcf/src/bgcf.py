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
"""Architecture"""
import mindspore.nn as nn
from mindspore import Parameter
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
from mindspore_gl.nn.conv.meanconv import MeanConv
from bgcfconv import AttenConv


class BGCF(nn.Cell):
    """
    BGCF architecture.

    Args:
        dataset_argv (list[int]): A list of the dataset argv.
        architect_argv (list[int]): A list of the model layer argv.
        activation (str): Activation function applied to the output of the layer, eg. 'relu'. Default: 'tanh'.
        neigh_drop_rate (list[float]): A list of the dropout ratio.
        num_user (int): The num of user.
        num_item (int): The num of item.
        input_dim (int): The feature dim.
    """

    def __init__(self,
                 dataset_argv,
                 architect_argv,
                 activation,
                 neigh_drop_rate,
                 num_user,
                 num_item,
                 input_dim):
        super(BGCF, self).__init__()

        self.ui_embed = Parameter(initializer("XavierUniform", [num_user + num_item, input_dim], dtype=mstype.float32))
        self.cast = P.Cast()
        self.tanh = P.Tanh()
        self.shape = P.Shape()
        self.split = P.Split(0, 2)
        self.gather = P.Gather()
        self.reshape = P.Reshape()
        self.concat_0 = P.Concat(0)
        self.concat_1 = P.Concat(1)

        (self.input_dim, self.num_user, self.num_item) = dataset_argv
        self.layer_dim = architect_argv

        self.gnew_agg_mean = MeanConv(self.input_dim, self.layer_dim,
                                      activation=activation, feat_drop=neigh_drop_rate[1])

        # self.gnew_agg_user = BGCFConv(self.input_dim, self.layer_dim, input_drop_out_rate=neigh_drop_rate[2])
        self.gnew_agg_user = AttenConv(self.input_dim, self.layer_dim, input_drop_out_rate=neigh_drop_rate[2])

        # self.gnew_agg_item = BGCFConv(self.input_dim, self.layer_dim, input_drop_out_rate=neigh_drop_rate[2])
        self.gnew_agg_item = AttenConv(self.input_dim, self.layer_dim, input_drop_out_rate=neigh_drop_rate[2])

        self.user_feature_dim = self.input_dim
        self.item_feature_dim = self.input_dim

        self.final_weight = Parameter(
            initializer("XavierUniform", [self.input_dim * 3, self.input_dim * 3], dtype=mstype.float32))

        self.raw_agg_funcs_user = MeanConv(self.input_dim, self.layer_dim,
                                           activation=activation, feat_drop=neigh_drop_rate[0])

        self.raw_agg_funcs_item = MeanConv(self.input_dim, self.layer_dim,
                                           activation=activation, feat_drop=neigh_drop_rate[0])

        self.tile = P.Tile()
        self.expand_dims = P.ExpandDims()

    def construct(self,
                  u_id,
                  pos_item_id,
                  neg_item_id,
                  pos_users,
                  pos_items,
                  u_group_nodes,
                  u_neighs,
                  u_gnew_neighs,
                  i_group_nodes,
                  i_neighs,
                  i_gnew_neighs,
                  neg_group_nodes,
                  neg_neighs,
                  neg_gnew_neighs,
                  neg_item_num):
        """Aggregate user and item embeddings"""
        all_user_embed = self.gather(self.ui_embed, self.concat_0((u_id, pos_users)), 0)

        num_samples = u_neighs.shape[1]
        num_bgcn_neigh = u_gnew_neighs.shape[1]
        ui_length = self.ui_embed.shape[0]

        u_group_nodes_dim = self.expand_dims(u_group_nodes, 1)
        i_group_nodes_dim = self.expand_dims(i_group_nodes, 1)
        neg_group_nodes_dim = self.expand_dims(neg_group_nodes, 1)

        u_group_nodes_1 = self.tile(u_group_nodes_dim, (1, num_samples))
        u_output_mean = self.raw_agg_funcs_user(self.ui_embed, u_group_nodes, u_neighs.flatten(),
                                                u_group_nodes_1.flatten(), ui_length, -1)

        u_group_nodes_2 = self.tile(u_group_nodes_dim, (1, num_bgcn_neigh))
        # u_group_cat_nodes = self.concat_1((u_group_nodes_1, u_group_nodes_2))
        u_neighs_cat_nodes = self.concat_1((u_neighs, u_gnew_neighs))
        u_output_from_gnew_mean = self.gnew_agg_mean(self.ui_embed, u_group_nodes, u_gnew_neighs.flatten(),
                                                     u_group_nodes_2.flatten(), ui_length, -1)
        # u_output_from_gnew_att = self.gnew_agg_user(self.ui_embed, u_group_nodes, u_neighs_cat_nodes.flatten(),
        # u_group_cat_nodes.flatten(), ui_length, -1)
        u_output_from_gnew_att = self.gnew_agg_user(self.ui_embed, u_group_nodes, u_neighs_cat_nodes)

        u_output = self.concat_1((u_output_mean, u_output_from_gnew_mean, u_output_from_gnew_att))
        all_user_rep = self.tanh(u_output)

        all_pos_item_embed = self.gather(self.ui_embed, self.concat_0((pos_item_id, pos_items)), 0)

        i_group_nodes_1 = self.tile(i_group_nodes_dim, (1, num_samples))
        i_output_mean = self.raw_agg_funcs_item(self.ui_embed, i_group_nodes, i_neighs.flatten(),
                                                i_group_nodes_1.flatten(), ui_length, -1)

        i_group_nodes_2 = self.tile(i_group_nodes_dim, (1, num_bgcn_neigh))
        i_output_from_gnew_mean = self.gnew_agg_mean(self.ui_embed, i_group_nodes, i_gnew_neighs.flatten(),
                                                     i_group_nodes_2.flatten(), ui_length, -1)

        # i_group_cat_nodes = self.concat_1((i_group_nodes_1, i_group_nodes_2))
        i_neighs_cat_nodes = self.concat_1((i_neighs, i_gnew_neighs))
        # i_output_from_gnew_att = self.gnew_agg_item(self.ui_embed, i_group_nodes, i_neighs_cat_nodes.flatten(),
        # i_group_cat_nodes.flatten(), ui_length, -1)
        i_output_from_gnew_att = self.gnew_agg_item(self.ui_embed, i_group_nodes, i_neighs_cat_nodes)

        i_output = self.concat_1((i_output_mean, i_output_from_gnew_mean, i_output_from_gnew_att))
        all_pos_item_rep = self.tanh(i_output)

        neg_item_embed = self.gather(self.ui_embed, neg_item_id, 0)

        neg_group_nodes_1 = self.tile(neg_group_nodes_dim, (1, num_samples))
        neg_output_mean = self.raw_agg_funcs_item(self.ui_embed, neg_group_nodes, neg_neighs.flatten(),
                                                  neg_group_nodes_1.flatten(), ui_length, -1)

        neg_group_nodes_2 = self.tile(neg_group_nodes_dim, (1, num_bgcn_neigh))
        neg_output_from_gnew_mean = self.gnew_agg_mean(self.ui_embed, neg_group_nodes, neg_gnew_neighs.flatten(),
                                                       neg_group_nodes_2.flatten(), ui_length, -1)

        # neg_group_cat_nodes = self.concat_1((neg_group_nodes_1, neg_group_nodes_2))
        neg_neighs_cat_nodes = self.concat_1((neg_neighs, neg_gnew_neighs))
        # neg_output_from_gnew_att = self.gnew_agg_item(self.ui_embed, neg_group_nodes, neg_neighs_cat_nodes.flatten(
        # ), neg_group_cat_nodes.flatten(), ui_length, -1)
        neg_output_from_gnew_att = self.gnew_agg_item(self.ui_embed, neg_group_nodes, neg_neighs_cat_nodes)

        neg_output = self.concat_1((neg_output_mean, neg_output_from_gnew_mean, neg_output_from_gnew_att))
        neg_output = self.tanh(neg_output)

        neg_output_shape = self.shape(neg_output)
        neg_item_rep = self.reshape(neg_output,
                                    (self.shape(neg_item_embed)[0], neg_item_num, neg_output_shape[-1]))

        return all_user_embed, all_user_rep, all_pos_item_embed, all_pos_item_rep, neg_item_embed, neg_item_rep
