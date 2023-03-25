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
"""Operations for Graph."""
from typing import List, Union, Tuple
import math
from enum import Enum
import numpy as np

import mindspore_gl.array_kernel as array_kernel
import mindspore_gl.dataloader.shared_numpy as shared_numpy
from .graph import BatchMeta, MindHomoGraph
from .utils import SharedArrayPool, ArrayPool


class BatchHomoGraph:
    """
    BatchHomoGraph, batch list of MindHomoGraph into a single MindHomoGraph with some batch_meta information.

    Inputs:
        - **graph_list** (List[MindHomoGraph]) - A list of MindHomoGraph.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindspore_gl.graph.ops import BatchHomoGraph
        >>> import numpy as np
        >>> from mindspore_gl.graph import MindHomoGraph
        >>> graph_list = []
        >>> for _ in range(5):
        ...     graph = MindHomoGraph()
        ...     edges = np.array([[0, 2, 2, 3, 4, 5, 5, 6], [1, 0, 1, 5, 3, 4, 6, 4]])
        ...     graph.set_topo_coo(edges)
        ...     graph.node_count = 7
        ...     graph.edge_count = 8
        ...     graph_list.append(graph)
        >>> batch_fn = BatchHomoGraph()
        >>> batch_graph = batch_fn(graph_list)
        >>> print(batch_graph.edge_count)
        40
    """

    def __init__(self):
        self.res_coo = None

    def __call__(self, graph_list: List[MindHomoGraph], **kwargs) -> MindHomoGraph:
        """determine coo_length"""
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
        # copy edge array
        node_offset = 0
        edge_offset = 0
        for graph in graph_list:
            self.res_coo[:, edge_offset: graph.edge_count + edge_offset] = graph.adj_coo + node_offset
            node_offset += graph.node_count
            edge_offset += graph.edge_count
        # Pack Result
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

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindspore_gl.graph.ops import BatchHomoGraph
        >>> import numpy as np
        >>> from mindspore_gl.graph import MindHomoGraph
        >>> graph_list = []
        >>> for _ in range(5):
        ...     graph = MindHomoGraph()
        ...     edges = np.array([[0, 2, 2, 3, 4, 5, 5, 6], [1, 0, 1, 5, 3, 4, 6, 4]])
        ...     graph.set_topo_coo(edges)
        ...     graph.node_count = 7
        ...     graph.edge_count = 8
        ...     graph_list.append(graph)
        >>> batch_fn = BatchHomoGraph()
        >>> batch_graph = batch_fn(graph_list)
        >>> unbatch_fn = UnBatchHomoGraph()
        >>> unbatch_graph = unbatch_fn(batch_graph)
        >>> print(unbatch_graph[0].edge_count)
        8
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
    Padding Mode, for graph and 2D array.

    - PadMode.CONST: padding the array into user specified shape.
    - PadMode.AUTO: auto generate the padding shape.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindspore_gl.graph import PadMode
        >>> const = PadMode.CONST
        >>> print(const.name, const.value)
        CONST 1
        >>> auto = PadMode.AUTO
        >>> print(auto.name, auto.value)
        AUTO 2
    """

    CONST = 1
    AUTO = 2


class PadDirection(Enum):
    """
    Padding Direction for 2d array specifically.

    - PadDirection.ROW: padding in the direction of the row.
    - PadDirection.COL: padding in the direction of the col.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindspore_gl.graph import PadDirection
        >>> row = PadDirection.ROW
        >>> print(row.name, row.value)
        ROW 1
        >>> col = PadDirection.COL
        >>> print(col.name, col.value)
        COL 2
    """

    ROW = 1
    COL = 2


class PadArray2d:
    r"""
    PadArray2d, specific pad operator for 2D array.

    .. warning::
        PadArray2d will reuse memory buffer to speedup pad operation.

    Args:
        dtype(numpy.dtype): To determine result's data type.
        direction(PadDirection): Pad direction for array, PadDirection.
            ROW means we will pad along axis=1, PadDirection.COl means we will pad along axis=0.
        fill_value(Union[float, int, optional]): Fill value for padded region. Default: None.
        reset_with_fill_value(bool, optional): PadArray2d will reuse memory buffer,
            you can set this value to False if you dont care about the padded value. Default: True.
        mode(PadMode, optional): Pad mode for array, if PadMode.CONST, this op will pad array to user-specific
            size. If PadMode.AUTO, this will choose padded result length according to input's length.
            The expected length can be calculated as
            .. math::
                length=2^{ceil\left ( \log_{2}{input\_length}  \right ) }
            Default: mindspore_gl.graph.PadMode.AUTO.
        size(Union[List, Tuple, optional]): User specific size for padding result. Default: None.
        use_shared_numpy(bool, optional): If we use SharedNDArray for speeding up inter process communication.
            This is recommended if you do feature collection and feature padding in child process and
            need inter process communication for graph feature. Default: False.

    Inputs:
        - **input_array** (numpy.array) - input numpy array for pad.

    Raises:
        ValueError: pad size should be provided when padding mode is PadMode.CONST.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindspore_gl.graph.ops import PadArray2d, PadMode, PadDirection
        >>> pad_op = PadArray2d(dtype=np.float32, mode=PadMode.CONST, direction=PadDirection.COL,
        ...                               size=(3, 1), fill_value=0)
        >>> node_list = np.array([[1]])
        >>> res = pad_op(node_list)
        >>> print(res)
        [[1.]
         [0.]
         [0.]]
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
            # Put Back To Memory Buffer
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
        # Put Back To Memory Buffer
        self.array_pool.put(target_size, memory_buffer)
        return memory_buffer

    def lazy(self, shape: Union[List, Tuple], **kwargs):
        """
        Lazy Array Pad, this will just determine padded result shape and return an empty array with target shape.

        Args:
            shape(Union[List, Tuple]): input array's shape for pad.
            kwargs(dict): config dict

                - **fill_value** (Union[int, float]): fill the padding array with value.

        Returns:
            memory_buffer(numpy.ndarray), an empty numpy array with target padded shape.

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
            # Put Back To Memory Buffer
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
        # Put Back To Memory Buffer
        self.array_pool.put(target_size, memory_buffer)
        return memory_buffer


class PadHomoGraph:
    r"""
    Pad MindHomoGraph, We pad graph by adding additional nodes and edges between these nodes. In short,
    :math:`PadHomoGraph(graph1) = BatchHomoGraph(graph1, fake\_graph)`
    node count and edge count in fake_graph is determined by user-specific parameters.

    Args:
        n_node(Union(int, None)): target graph's node count. Default: None.
        n_edge(Union(int, None)): target graph's edge count. Default: None.
        mode(PadMode): Pad mode, if PadMode.CONST, target graph will have n_node nodes and n_edge edges. If PadMode.AUTO
            target graph's node_count and edge_count is calculated according to input graph's size by
            :math:`n\_node = 2^{ceil(log2(input\_graph.node\_count))}` ,
            :math:`n\_edge = 2^{ceil(log2(input\_graph.edge\_count))}` . Default: PadMode.AUTO.
        csr(bool): Is the csr graph. Default: False.

    Inputs:
        - **graph** (MindHomoGraph) - input graph.

    Outputs:
        - MindHomoGraph, padded graph.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindspore_gl.graph.ops import BatchHomoGraph, PadHomoGraph, PadMode
        >>> import numpy as np
        >>> from mindspore_gl.graph.graph import MindHomoGraph
        >>> graph_list = []
        >>> for _ in range(1):
        ...     graph = MindHomoGraph()
        ...     edges = np.array([[0, 2, 2, 3, 4, 5, 5, 6], [1, 0, 1, 5, 3, 4, 6, 4]])
        ...     graph.set_topo_coo(edges)
        ...     graph.node_count = 7
        ...     graph.edge_count = 8
        ...     graph_list.append(graph)
        >>> batch_fn = BatchHomoGraph()
        >>> batch_graph = batch_fn(graph_list)
        >>> n_node = graph.node_count + 1
        >>> n_edge = graph.edge_count + 30
        >>> pad_graph_op = PadHomoGraph(mode=PadMode.CONST, n_node=n_node, n_edge=n_edge)
        >>> pad_res = pad_graph_op(batch_graph)
        >>> print(pad_res[0].edge_count, pad_res[1].edge_count)
        8   30
        >>> print(pad_res[0].node_count, pad_res[1].node_count)
        7   1
    """

    def __init__(self, n_node=None, mode=PadMode.AUTO, n_edge=None, csr=False):
        if mode == PadMode.CONST:
            assert n_edge is not None and n_node is not None, \
                "n_node and n_edge should be given when padding with CONST Mode"

        self.n_node = n_node
        self.mode = mode
        self.n_edge = n_edge
        self.batch_op = BatchHomoGraph()
        self.csr = csr

    def __call__(self, graph: MindHomoGraph, **kwargs) -> MindHomoGraph:
        """
        Do pad operation.
        """
        # Check Input Graph is Valid To Pad
        res_graph = MindHomoGraph()
        if self.mode is PadMode.CONST:
            assert graph.edge_count < self.n_edge, \
                "Given graph is too large for the given padding"
        if graph.is_batched:
            if self.mode == PadMode.CONST:
                # No Need To Pad
                if graph.edge_count == self.n_edge:
                    return graph
                # Determine Padded Graph
                if self.csr:
                    pad_graph_coo = generate_fill_array(graph.adj_coo, (2, self.n_edge), self.n_node - 1)
                else:
                    pad_graph_coo = np.full([2, self.n_edge - graph.edge_count], self.n_node - 1, dtype=np.int32)
                pad_graph = MindHomoGraph()
                pad_graph.adj_coo = pad_graph_coo
                pad_graph.node_count = self.n_node - graph.node_count
                pad_graph.edge_count = self.n_edge - graph.edge_count
                # Pad Graph
                res_graph.adj_coo = np.concatenate([graph.adj_coo, pad_graph.adj_coo], axis=1)
                res_graph_graph_nodes = np.concatenate([graph.batch_meta.graph_nodes, np.array([self.n_node],
                                                                                               dtype=np.int32)])
                res_graph_graph_edges = np.concatenate([graph.batch_meta.graph_edges, np.array([self.n_edge],
                                                                                               dtype=np.int32)])
                res_graph.batch_meta = BatchMeta(graph_nodes=res_graph_graph_nodes, graph_edges=res_graph_graph_edges)
                res_graph.edge_count = self.n_edge
                res_graph.node_count = self.n_node
                return res_graph
            # No Need To Pad
            if graph.edge_count == 1 << math.ceil(math.log2(graph.edge_count)):
                return graph
            # Determine Pad Graph
            edge_bucket_length = math.ceil(math.log2(graph.edge_count))
            padded_graph_edge_count = (1 << edge_bucket_length) - graph.edge_count
            padded_graph_node_count = (1 << math.ceil(math.log2(graph.node_count))) - graph.node_count
            pad_graph = MindHomoGraph()
            pad_value = (1 << math.ceil(math.log2(graph.node_count))) - 1
            pad_graph.adj_coo = np.full([2, padded_graph_edge_count], pad_value, dtype=np.int32)
            pad_graph.node_count = padded_graph_node_count
            pad_graph.edge_count = padded_graph_edge_count

            # Pad Graph
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
            # No Need To Pad
            if graph.edge_count == self.n_edge:
                return graph
            # Determine Pad Graph
            pad_graph_coo = np.full([2, self.n_edge - graph.edge_count], self.n_node - 1, dtype=np.int32)
            pad_graph = MindHomoGraph()
            pad_graph.adj_coo = pad_graph_coo
            pad_graph.node_count = self.n_node - graph.node_count
            pad_graph.edge_count = self.n_edge - graph.edge_count
            return self.batch_op([graph, pad_graph])
        # No Need To Pad
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

class PadCsrEdge:
    """
        PadCsrEdge, specific pad operator for coo edges. After padding, the shape of the coo edge index to the
        csr indices and indptr becomes unified.

        .. warning::
            PadArray2d will reuse memory buffer to speedup pad operation.

        Args:
            pad_nodes(int): nodes numbers of the graph.
            reset_with_fill_value(bool): PadArray2d will reuse memory buffer,
                you can set this value to False if you dont care about the padded value. Default: True.
            length(int): User specific length for padding result. Default: None.
            mode(PadMode): Pad mode for array, if PadMode.CONST, this op will pad array to user-specific size.
                If PadMode.AUTO, this will choose padded result length according to input's length.
                The expected length can be calculated as 2^ceil(log2(input_length)).
                Default: mindspore_gl.graph.PadMode.AUTO.
            use_shared_numpy(bool): If we use SharedNDArray for speeding up inter process communication.
                This is recommended if you do feature collection and feature padding in child process and
                need inter process communication for graph feature. Default: False.

        Inputs:
            - **input_array** (numpy.array) - input numpy array for pad.

        Raises:
            ValueError: pad length should be provided when padding mode is PadMode.CONST.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            import numpy as np
            >>> from mindspore_gl.graph import PadCsrEdge, PadMode
            >>> node_pad = 10
            >>> origin_edge_index = np.array([[0, 1, 2, 4],
            ...                               [2, 3, 1, 1]])
            >>> pad_length = 20
            >>> pad_op = PadCsrEdge(node_pad, length=pad_length, mode=PadMode.CONST)
            >>> res = pad_op(origin_edge_index)
            >>> print(res)
            [[0 1 2 4 5 6 7 8 5 6 7 8 5 6 7 8 5 6 7 8]
             [2 3 1 1 5 6 7 8 6 7 8 5 7 8 5 6 8 5 6 7]]
    """

    def __init__(self, pad_nodes, reset_with_fill_value=True, length=None, mode=PadMode.AUTO,
                 use_shared_numpy=False):
        self.pad_nodes = pad_nodes
        self.pad_mode = mode
        self.homo_batch = BatchHomoGraph()
        self.reset_with_fill_value = reset_with_fill_value
        self.use_shared_numpy = use_shared_numpy
        self.size = (2, length)
        if self.use_shared_numpy:
            self.array_pool = SharedArrayPool()
        else:
            self.array_pool = ArrayPool()

        if mode == PadMode.CONST:
            if self.use_shared_numpy:
                memory_buffer = shared_numpy.SharedNDArray.from_shape(self.size, dtype=np.int32)
                self.array_pool.put(self.size, memory_buffer)
            else:
                memory_buffer = np.zeros(self.size, dtype=np.int32)
                self.array_pool.put(self.size, memory_buffer)

    def __call__(self, input_array):
        """
        Pad Array
        """
        if self.pad_mode == PadMode.CONST:
            fill_array = generate_fill_array(input_array, self.size, self.pad_nodes)
            memory_buffer = self.array_pool.pop(self.size)
            # If Memory Buffer Is None, It's Definitely SharedNDArray
            if memory_buffer is None:
                memory_buffer = shared_numpy.SharedNDArray.from_shape(self.size, dtype=np.int32)
                self.array_pool.put(self.size, memory_buffer)

            memory_buffer[:, :input_array.shape[1]] = input_array
            if self.reset_with_fill_value:
                memory_buffer[:, input_array.shape[1]:] = fill_array
            # Put Back To Memory Buffer
            self.array_pool.put(self.size, memory_buffer)
            return memory_buffer
        bucket_length = math.ceil(math.log2(input_array.shape[1]))
        target_size = [2, 1 << bucket_length]
        fill_value = generate_fill_array(input_array, target_size, self.pad_nodes)

        memory_buffer = self.array_pool.pop(target_size)
        if memory_buffer is None:
            memory_buffer = shared_numpy.SharedNDArray.from_shape(target_size, np.int32)
            self.array_pool.put(target_size, memory_buffer)

        memory_buffer[:, :input_array.shape[1]] = input_array
        if self.reset_with_fill_value:
            memory_buffer[:, input_array.shape[1]:] = fill_value
        # Put Back To Memory Buffer
        self.array_pool.put(target_size, memory_buffer)
        return memory_buffer

def generate_fill_array(input_array, size, pad_nodes):
    """generate the fill array"""
    start = np.max(input_array) + 1
    end = pad_nodes - 1
    mini_length = size[1] - input_array.shape[1]
    fill_array = np.array([np.arange(start, end), np.arange(start, end)])
    add_bias = 0
    while fill_array.shape[1] < mini_length:
        add_array_0 = np.arange(start, end)
        add_array_1 = np.arange(start, end)
        add_array_1 = np.append(add_array_1[add_bias + 1:], add_array_1[:add_bias + 1])
        add_array = np.array([add_array_0, add_array_1])
        fill_array = np.concatenate((fill_array, add_array), axis=1)
        add_bias += 1
        if add_bias == end - start:
            break
    fill_array = fill_array[:, :mini_length]
    return fill_array
