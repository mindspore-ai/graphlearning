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

"""Operations for Graph"""
from typing import List, Union, Tuple
import math
from enum import Enum
import numpy as np

import mindspore_gl.array_kernel as array_kernel
from mindspore_gl.dataloader import shared_numpy
from .graph import BatchMeta, MindHomoGraph
from .utils import SharedArrayPool, ArrayPool


class BatchHomoGraph:
    """
    BatchHomoGraph, batch list of MindHomoGraph into a single MindHomoGraph with some batch_meta information.
    """

    def __init__(self):
        self.res_coo = None

    def __call__(self, graph_list: List[MindHomoGraph], **kwargs) -> MindHomoGraph:
        ########################
        # determine coo_length
        #########################
        total_edge_count = 0
        total_node_count = 0
        graph_nodes = np.zeros([len(graph_list) + 1], dtype=np.int32)
        graph_edges = np.zeros([len(graph_list) + 1], dtype=np.int32)
        for idx, graph in enumerate(graph_list):
            total_edge_count += graph.edge_count
            total_node_count += graph.node_count
            graph_edges[idx + 1] = total_edge_count
            graph_nodes[idx + 1] = total_node_count

        if self.res_coo is None or self.res_coo.shape[1] < total_edge_count:
            del self.res_coo
            self.res_coo = np.zeros([2, total_edge_count], dtype=np.int32)
        ###########################
        # copy edge array
        ########################
        node_offset = 0
        edge_offset = 0
        for graph in graph_list:
            self.res_coo[:, edge_offset: graph.edge_count + edge_offset] = graph.adj_coo + node_offset
            node_offset += graph.node_count
            edge_offset += graph.edge_count
        ######################################
        # Pack Result
        ######################################
        res_graph = MindHomoGraph()
        res_graph.set_topo_coo(np.copy(self.res_coo[:, :total_edge_count]))
        res_graph.node_count = total_node_count
        res_graph.edge_count = total_edge_count
        batch_meta = BatchMeta(graph_nodes=graph_nodes, graph_edges=graph_edges)
        res_graph.batch_meta = batch_meta
        return res_graph


class UnBatchHomoGraph:
    """
    Return list of MindHomoGraph from a Batched MindHomoGraph.
    """

    def __init__(self):
        pass

    def __call__(self, graph: MindHomoGraph, **kwargs) -> List[MindHomoGraph]:
        assert graph.is_batched, "UnBatchHomoGraph can only be operated on batched_graph"
        res: List[MindHomoGraph] = []
        for idx in range(graph.batch_meta.graph_count):
            res.append(graph[idx])
        return res


class PadMode(Enum):
    """
    Padding Mode, for graph and 2d array.
    """
    CONST = 1
    AUTO = 2


class PadDirection(Enum):
    """
    Padding Direction for 2d array specifically.

    """
    ROW = 1
    COL = 2


class PadArray2d:
    """
    PadArray2d, specific pad operator for 2d array.

    .. warning::
        PadArray2d will reuse memory buffer to speedup pad operation.

    Args:
        dtype(numpy.dtype): To determine result's data type.
        direction(PadDirection): Pad direction for array, PadDirection.
            ROW means we will pad along axis=1, PadDirection.COl means we will pad along axis=0.
        fill_value(Union[float, int, None]): Fill value for padded region.
        reset_with_fill_value(bool): PadArray2d will reuse memory buffer,
            you can set this value to False if you dont care about the padded value.
        mode(PadMode): Pad mode for array, if PadMode.CONST, this op will pad array to user-specific size.
            If PadMode.AUTO, this will choose padded result length according to input's length.
            The expected length can be calculated as 2^ceil(log2(input_length)).
        size(Union[List, Tuple]): User specific size for padding result.
        use_shared_numpy(bool): If we use SharedNDArray for speeding up inter process communication.
            This is recommended if you do feature collection and feature padding in child process and
            need inter process communication for graph feature.
    """

    def __init__(self, dtype, direction, fill_value=None, reset_with_fill_value=True, mode=PadMode.AUTO, size=None,
                 use_shared_numpy=False):
        if mode == PadMode.CONST:
            assert size is not None and dtype is not None and fill_value is not None, \
                "pad size should be provided when padding mode is PadMode.CONST"
        self.pad_mode = mode
        self.pad_direction = direction
        self.fill_value = fill_value
        self.dtype = dtype
        self.homo_batch = BatchHomoGraph()
        self.reset_with_fill_value = reset_with_fill_value
        self.use_shared_numpy = use_shared_numpy
        self.size = size
        if self.use_shared_numpy:
            self.array_pool = SharedArrayPool()
        else:
            self.array_pool = ArrayPool()

        if mode == PadMode.CONST:
            if self.use_shared_numpy:
                memory_buffer = shared_numpy.SharedNDArray.from_shape(size, dtype=dtype)
                self.array_pool.put(size, memory_buffer)
            else:
                memory_buffer = np.zeros(size, dtype=dtype)
                self.array_pool.put(size, memory_buffer)

    def __call__(self, input_array, **kwargs):
        """
        Pad Array

        Args:
            input_array(numpy.array): input numpy array for pad

        Returns:
            numpy.array, padded array
        """
        fill_value = kwargs.get("fill_value", None)
        if self.pad_mode == PadMode.CONST:
            memory_buffer = self.array_pool.pop(self.size)
            # If Memory Buffer Is None, It's Definitely SharedNDArray
            if memory_buffer is None:
                memory_buffer = shared_numpy.SharedNDArray.from_shape(self.size, dtype=self.dtype)
                self.array_pool.put(self.size, memory_buffer)

            if self.pad_direction == PadDirection.ROW:
                memory_buffer[:, :input_array.shape[1]] = input_array
                if self.reset_with_fill_value:
                    memory_buffer[:, input_array.shape[1]:] = self.fill_value
            else:
                if input_array.dtype == np.float32 and self.dtype == np.float32:
                    array_kernel.float_2d_array_col_copy(memory_buffer, input_array)
                else:
                    memory_buffer[:input_array.shape[0]] = input_array
                if self.reset_with_fill_value:
                    memory_buffer[input_array.shape[0]:] = self.fill_value
            ##########################
            # Put Back To Memory Buffer
            ##########################
            self.array_pool.put(self.size, memory_buffer)
            return memory_buffer
        memory_buffer = None
        target_size = None
        if self.pad_direction == PadDirection.ROW:
            bucket_length = math.ceil(math.log2(input_array.shape[1]))
            target_size = [input_array.shape[0], 1 << bucket_length]
            if fill_value is None:
                fill_value = self.fill_value or (1 << bucket_length) - 1

            memory_buffer = self.array_pool.pop(target_size)
            if memory_buffer is None:
                memory_buffer = shared_numpy.SharedNDArray.from_shape(target_size, self.dtype)
                self.array_pool.put(target_size, memory_buffer)

            memory_buffer[:, :input_array.shape[1]] = input_array
            if self.reset_with_fill_value:
                memory_buffer[:, input_array.shape[1]:] = fill_value
        else:
            bucket_length = math.ceil(math.log2(input_array.shape[0]))
            target_size = [1 << bucket_length, input_array.shape[1]]

            if fill_value is None:
                fill_value = self.fill_value or (1 << bucket_length) - 1
            memory_buffer = self.array_pool.pop(target_size)

            if memory_buffer is None:
                memory_buffer = shared_numpy.SharedNDArray.from_shape(target_size, self.dtype)
                self.array_pool.put(target_size, memory_buffer)

            memory_buffer[:input_array.shape[0]] = input_array
            if self.reset_with_fill_value:
                memory_buffer[input_array.shape[0]:] = fill_value
        ##########################
        # Put Back To Memory Buffer
        ##########################
        self.array_pool.put(target_size, memory_buffer)
        return memory_buffer

    def lazy(self, shape: Union[List, Tuple], **kwargs):
        """
        Lazy Array Pad, this will just determine padded result shape and return an empty array with target shape.

        Args:
            shape( Union[List, Tuple]): input array's shape for pad.

        Returns:
            memory_buffer(numpy.array), an empty numpy array with target padded shape.

        """
        fill_value = kwargs.get("fill_value", None)
        if self.pad_mode == PadMode.CONST:
            memory_buffer = self.array_pool.pop(self.size)
            # If Memory Buffer Is None, It's Definitely SharedNDArray
            if memory_buffer is None:
                memory_buffer = shared_numpy.SharedNDArray.from_shape(self.size, dtype=self.dtype)

            if self.reset_with_fill_value:

                if self.pad_direction == PadDirection.ROW:
                    memory_buffer[:, shape[1]:] = self.fill_value
                else:
                    memory_buffer[shape[0]:] = self.fill_value
            ##########################
            # Put Back To Memory Buffer
            ##########################
            self.array_pool.put(self.size, memory_buffer)
            return memory_buffer
        memory_buffer = None
        target_size = None
        if self.pad_direction == PadDirection.ROW:
            bucket_length = math.ceil(math.log2(shape[1]))
            target_size = [shape[0], 1 << bucket_length]
            if fill_value is None:
                fill_value = self.fill_value or (1 << bucket_length) - 1

            memory_buffer = self.array_pool.pop(target_size)
            if memory_buffer is None:
                memory_buffer = shared_numpy.SharedNDArray.from_shape(target_size, self.dtype)

            if self.reset_with_fill_value:
                memory_buffer[:, shape[1]:] = fill_value
        else:
            bucket_length = math.ceil(math.log2(shape[0]))
            target_size = [1 << bucket_length, shape[1]]

            if fill_value is None:
                fill_value = self.fill_value or (1 << bucket_length) - 1
            memory_buffer = self.array_pool.pop(target_size)

            if memory_buffer is None:
                memory_buffer = shared_numpy.SharedNDArray.from_shape(target_size, self.dtype)

            if self.reset_with_fill_value:
                memory_buffer[shape[0]:] = fill_value
        ##########################
        # Put Back To Memory Buffer
        ##########################
        self.array_pool.put(target_size, memory_buffer)
        return memory_buffer


class PadHomoGraph:
    """
    Pad MindHomoGraph, We pad graph by adding additional nodes and edges between these nodes. In short,
    PadHomoGraph(graph1) = BatchHomoGraph(graph1, fake_graph)
    node count and edge count in fake_graph is determined by user-specific parameters

    Args:
        n_node(Union(int, None)): target graph's node count
        n_edge(Union(int, None)): target graph's edge count
        mode(PadMode): Pad mode, if PadMode.CONST, target graph will have n_node nodes and n_edge edges. If PadMode.AUTO
            target graph's node_count and edge_count is calculated according to input graph's size by
            n_node = 2^ceil(log2(input_graph.node_count)),
            n_edge = 2^ceil(log2(input_graph.edge_count))

    """

    def __init__(self, n_node=None, mode=PadMode.AUTO, n_edge=None):
        if mode == PadMode.CONST:
            assert n_edge is not None and n_node is not None, \
                "n_node and n_edge should be given when padding with CONST Mode"

        self.n_node = n_node
        self.mode = mode
        self.n_edge = n_edge
        self.batch_op = BatchHomoGraph()

    def __call__(self, graph: MindHomoGraph, **kwargs) -> MindHomoGraph:
        """
        Do pad operation.

        Args:
            graph(MindHomoGraph): input graph

        Returns:
            MindHomoGraph, padded graph
        """
        ####################################
        # Check Input Graph is Valid To Pad
        ####################################
        res_graph = MindHomoGraph()
        if self.mode is PadMode.CONST:
            assert graph.edge_count < self.n_edge, \
                "Given graph is too large for the given padding"
        if graph.is_batched:
            if self.mode == PadMode.CONST:
                ###########################
                # No Need To Pad
                ###########################
                if graph.edge_count == self.n_edge:
                    return graph
                ####################################
                # Determine Padded Graph
                ####################################
                pad_graph_coo = np.full([2, self.n_edge - graph.edge_count], self.n_node - 1, dtype=np.int32)
                pad_graph = MindHomoGraph()
                pad_graph.adj_coo = pad_graph_coo
                pad_graph.node_count = self.n_node - graph.node_count
                pad_graph.edge_count = self.n_edge - graph.edge_count
                ####################################
                # Pad Graph
                ####################################
                res_graph.adj_coo = np.concatenate([graph.adj_coo, pad_graph.adj_coo], axis=1)
                res_graph_graph_nodes = np.concatenate([graph.batch_meta.graph_nodes, np.array([self.n_node],
                                                                                               dtype=np.int32)])
                res_graph_graph_edges = np.concatenate([graph.batch_meta.graph_edges, np.array([self.n_edge],
                                                                                               dtype=np.int32)])
                res_graph.batch_meta = BatchMeta(graph_nodes=res_graph_graph_nodes, graph_edges=res_graph_graph_edges)
                res_graph.edge_count = self.n_edge
                res_graph.node_count = self.n_node
                return res_graph
            ###########################
            # No Need To Pad
            ###########################
            if graph.edge_count == 1 << math.ceil(math.log2(graph.edge_count)):
                return graph
            #############################
            # Determine Pad Graph
            #############################
            edge_bucket_length = math.ceil(math.log2(graph.edge_count))
            padded_graph_edge_count = (1 << edge_bucket_length) - graph.edge_count
            padded_graph_node_count = (1 << math.ceil(math.log2(graph.node_count))) - graph.node_count
            pad_graph = MindHomoGraph()
            pad_value = (1 << math.ceil(math.log2(graph.node_count))) - 1
            pad_graph.adj_coo = np.full([2, padded_graph_edge_count], pad_value, dtype=np.int32)
            pad_graph.node_count = padded_graph_node_count
            pad_graph.edge_count = padded_graph_edge_count

            ################################
            # Pad Graph
            ################################
            res_graph.adj_coo = np.concatenate([graph.adj_coo, pad_graph.adj_coo], axis=1)
            res_graph_graph_nodes = np.concatenate([graph.batch_meta.graph_nodes,
                                                    np.array([1 << math.ceil(math.log2(graph.node_count))],
                                                             dtype=np.int32)])
            res_graph_graph_edges = np.concatenate([graph.batch_meta.graph_edges,
                                                    np.array([1 << edge_bucket_length], dtype=np.int32)])
            res_graph.batch_meta = BatchMeta(graph_nodes=res_graph_graph_nodes, graph_edges=res_graph_graph_edges)
            res_graph.edge_count = graph.edge_count + pad_graph.edge_count
            res_graph.node_count = graph.node_count + pad_graph.node_count

            return res_graph
        if self.mode == PadMode.CONST:
            ###############################
            # No Need To Pad
            ###############################
            if graph.edge_count == self.n_edge:
                return graph
            ############################
            # Determine Pad Graph
            ############################
            pad_graph_coo = np.full([2, self.n_edge - graph.edge_count], self.n_node - 1, dtype=np.int32)
            pad_graph = MindHomoGraph()
            pad_graph.adj_coo = pad_graph_coo
            pad_graph.node_count = self.n_node - graph.node_count
            pad_graph.edge_count = self.n_edge - graph.edge_count
            return self.batch_op([graph, pad_graph])
        ###########################
        # No Need To Pad
        ###########################
        if graph.edge_count == 1 << math.ceil(math.log2(graph.edge_count)):
            return graph

        edge_bucket_length = math.ceil(math.log2(graph.edge_count))
        padded_graph_edge_count = (1 << edge_bucket_length) - graph.edge_count
        padded_graph_node_count = (1 << math.ceil(math.log2(graph.node_count))) - graph.node_count
        pad_graph = MindHomoGraph()
        pad_value = (1 << math.ceil(math.log2(graph.node_count))) - 1
        pad_graph.adj_coo = np.full([2, padded_graph_edge_count], pad_value, dtype=np.int32)
        pad_graph.node_count = padded_graph_node_count
        pad_graph.edge_count = padded_graph_edge_count
        return self.batch_op([graph, pad_graph])


class UnPadHomoGraph:
    """Empty placeholder"""

    def __init__(self):
        pass

    def __call__(self, graph: MindHomoGraph, **kwargs) -> MindHomoGraph:
        pass
