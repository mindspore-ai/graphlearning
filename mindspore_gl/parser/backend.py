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
"""Backend for transformation."""
# pylint: disable=bad-continuation,unused-argument
import ast
from typing import List
import numpy as np

from .vectorization import VectorizationType
from .constants import GATHER_OP, ZEROS_OP, RESHAPE_OP, SHAPE_OP, \
                       SCATTER_SRC_IDX, SCATTER_DST_IDX, \
                       SCATTER_EDGE_SUBGRAPH_IDX, SCATTER_VER_SUBGRAPH_IDX,\
                       SCATTER_MIN_OP, SCATTER_MAX_OP, SCATTER_ADD_OP, \
                       SRC_IDX, DST_IDX, VER_SUBGRAPH_IDX, EDGE_SUBGRAPH_IDX, \
                       GRAPH_MASK, N_GRAPHS, N_NODES, GRAPH_FIELD_NAMES, \
                       BACKEND_NAME, FILL_OP, MASKED_FILL_OP, IS_INF_OP, INDICES, INDPTR, CSR_REDUCE_SUM_OP, \
                       INDICES_BACKWARD, INDPTR_BACKWARD


def set_backend(bk_name: str):
    """
    Set a backend.

    Args:
        bk_name (str): Backend name.
    """
    global BACKEND_NAME
    BACKEND_NAME = bk_name


def backend() -> str:
    """
    Get the backend name.

    Returns:
        str, backend name.
    """
    return BACKEND_NAME


class Backend:
    """Backend parent class."""
    csr = False
    backward = False

    def __init__(self) -> None:
        pass

    def init_graph_indices(self) -> List[ast.AST]:
        """
        Init graph indices.

        Returns:
            List[ast.AST], list of graph indices.
        """
        return []

    def create_gather_node(self, node, old_type) -> ast.AST:
        """
        Create a gather node.

        Args:
            node (ast.AST): Origin node.
            old_type (VectorizationType): Determine whether src_idx or dst_idx
                is used for gather.

        Returns:
            ast.AST, call after transformation.
        """
        call = ast.Call()

        if not self.csr:
            call.func = ast.Name(GATHER_OP)
            call.args = [node,
                         ast.Name(id=SRC_IDX, ctx=ast.Load())
                         if old_type == VectorizationType.SRC
                         else ast.Name(DST_IDX, ctx=ast.Load()),
                         ast.Constant(0)]
        else:
            if self.backward:
                call.func = ast.Name("self.CSR_BACKWARD_GATHER")
                call.args = [node,
                             ast.Name(id=INDICES, ctx=ast.Load()),
                             ast.Constant(0),
                             ast.Name(id=INDPTR_BACKWARD, ctx=ast.Load()),
                             ast.Name(id=INDICES_BACKWARD, ctx=ast.Load())
                             ]
            else:
                call.func = ast.Name(GATHER_OP)
                call.args = [node,
                             ast.Name(id=INDICES, ctx=ast.Load()),
                             ast.Constant(0)
                             ]
        call.keywords = []
        return call

    def inline_attribute_setter(self, node: ast.Expr) -> ast.AST:
        """
        Inline attribute setter for InlineAttributeSetter in ast_transformer.

        Args:
            node (ast.Expr): Origin node.

        Returns:
            ast.AST, call node after transformation.
        """
        call_node = node.value
        args = call_node.args
        dic = args[0]
        assign_node = ast.Assign(targets=[ast.Tuple(
            elts=[ast.Name(k.value
                           if isinstance(k, ast.Constant)
                           else k.s, ctx=ast.Store()) for k in dic.keys],
            ctx=ast.Store())],
                                 value=ast.List(elts=dic.values, ctx=ast.Load()))
        return assign_node

    def init_ops(self) -> List[ast.AST]:
        """Init ops."""
        raise NotImplementedError()

    def init_intermediates(self, graph_type) -> List[ast.AST]:
        """Init intermediates."""
        raise NotImplementedError()

    def transform_agg_func(self,
                           node: ast.AST,
                           enclosing_block: ast.AST,
                           insert_stmt_cb,
                           scatter_name,
                           is_avg) -> ast.Call:
        """Transform the aggregation function."""
        raise NotImplementedError()


class MindSporeBackend(Backend):
    """MindSpore Backend."""

    def __init__(self) -> None:
        super().__init__()
        self.snapshot_id = 0
        self.pos_inf = np.inf
        self.neg_inf = -np.inf

    def get_next_snapshot_id(self):
        """
        Get next snapshot id.

        Returns:
            int, snapshot id.
        """
        self.snapshot_id += 1
        return self.snapshot_id

    def init_ops(self) -> List[ast.AST]:
        """
        Init ops.

        Insert statements:
            SCATTER_ADD = ms.ops.TensorScatterAdd()
            SCATTER_MAX = ms.ops.TensorScatterMax()
            SCATTER_MIN = ms.ops.TensorScatterMin()
            GATHER = ms.ops.Gather()
            ZEROS = ms.ops.Zeros()
            SHAPE = ms.ops.Shape()
            RESHAPE = ms.ops.Reshape()

        Returns:
            List[ast.AST], ops list.
        """
        op_list = [self.init_op(SCATTER_ADD_OP, "TensorScatterAdd"),
                   self.init_op(SCATTER_MAX_OP, "TensorScatterMax"),
                   self.init_op(SCATTER_MIN_OP, "TensorScatterMin"),
                   self.init_op(GATHER_OP, "Gather"),
                   self.init_op(ZEROS_OP, "Zeros"),
                   self.init_op(FILL_OP, "Fill"),
                   self.init_op(MASKED_FILL_OP, "MaskedFill"),
                   self.init_op(IS_INF_OP, "IsInf"),
                   self.init_op(SHAPE_OP, "Shape"),
                   self.init_op(RESHAPE_OP, "Reshape")]
        if self.csr:
            csr_op = [self.init_op(CSR_REDUCE_SUM_OP, "operations._csr_ops.CSRReduceSum")]
            op_list.extend(csr_op)
        return op_list

    def init_intermediates(self, graph_type) -> List[ast.AST]:
        """
        Insert intermediate statements

        If graph_type is \"Graph\", insert statements:
            scatter_src_idx = reshape(src_idx, (shape(src_idx)[0], 1))
            scatter_dst_idx = reshape(dst_idx, (shape(dst_idx)[0], 1))
        Else if graph_type is \"BatchedGraph\", insert statements:
            scatter_src_idx = reshape(src_idx, (shape(src_idx)[0], 1))
            scatter_dst_idx = reshape(dst_idx, (shape(dst_idx)[0], 1))
            scatter_ver_subgraph_idx = reshape(ver_subgraph_idx,
                                            (shape(ver_subgraph_idx)[0], 1))
            scatter_edge_subgraph_idx = reshape(edge_subgraph_idx,
                                            (shape(edge_subgraph_idx)[0], 1))
            n_graphs = SHAPE(graph_mask)[0]

        Args:
            graph_type (str): graph type, can be Graph or BatchedGraph.

        Returns:
            List[ast.AST], intermediate list.

        Raises:
            SyntaxError: be raised if graph type not support.
        """
        if graph_type == "Graph":
            if self.csr:
                return []
            return [
                self.invoke_init_intermediate(SCATTER_SRC_IDX, SRC_IDX),
                self.invoke_init_intermediate(SCATTER_DST_IDX, DST_IDX),
            ]
        if graph_type == "BatchedGraph":
            if self.csr:
                return [
                        self.invoke_init_intermediate(SCATTER_VER_SUBGRAPH_IDX,
                                                      VER_SUBGRAPH_IDX),
                        self.invoke_init_intermediate(SCATTER_EDGE_SUBGRAPH_IDX,
                                                      EDGE_SUBGRAPH_IDX),
                        self.invoke_get_shape(N_GRAPHS, GRAPH_MASK),
                        ]
            return [
                    self.invoke_init_intermediate(SCATTER_SRC_IDX, SRC_IDX),
                    self.invoke_init_intermediate(SCATTER_DST_IDX, DST_IDX),
                    self.invoke_init_intermediate(SCATTER_VER_SUBGRAPH_IDX,
                                                  VER_SUBGRAPH_IDX),
                    self.invoke_init_intermediate(SCATTER_EDGE_SUBGRAPH_IDX,
                                                  EDGE_SUBGRAPH_IDX),
                    self.invoke_get_shape(N_GRAPHS, GRAPH_MASK),
                    ]
        raise SyntaxError("Graph type not support.")

    def transform_agg_func(self,
                           node: ast.AST,
                           enclosing_block: ast.AST,
                           insert_stmt_cb,
                           scatter_name,
                           is_avg) -> ast.Call:
        """
        Transform the aggregation functions.

        Args:
            node (ast.AST): The origin node.
            enclosing_block (ast.AST): The enclosing block for the node.
            insert_stmt_cb (Function): Insert statement callback.
            scatter_name (str): Scatter name, in SCATTER_ADD, SCATTER_MAX
                and SCATTER_MIN.
            is_avg (bool): Whether it is a avg operation.

        Returns:
            ast.AST, call node after transformation.

        Raises:
            SyntaxError: be raised if scatter_name not correct.
        """
        if scatter_name not in (SCATTER_ADD_OP,
                                SCATTER_MAX_OP,
                                SCATTER_MIN_OP):
            raise SyntaxError(f"scatter name {scatter_name} not support yet.")
        if self.csr:
            call = self.invoke_csr_reduce(node.args[0],
                                        enclosing_block,
                                        insert_stmt_cb,
                                        scatter_name,
                                        N_NODES,
                                        SCATTER_DST_IDX,
                                        is_avg)
        else:
            call = self.invoke_scatter_op(node.args[0],
                                        enclosing_block,
                                        insert_stmt_cb,
                                        scatter_name,
                                        N_NODES,
                                        SCATTER_DST_IDX,
                                        is_avg)
        return call

    def transform_dot_func(self,
                           node: ast.AST,
                           enclosing_block: ast.AST,
                           insert_stmt_cb) -> ast.Call:
        """
        Transform the dot functions.

        Args:
            node (ast.AST): The origin node.
            enclosing_block (ast.AST): The enclosing block for the node.
            insert_stmt_cb (Function): Insert statement callback.

        Returns:
            ast.AST, call node after transformation.
        """
        call = ast.Call()
        call.func = ast.Call(func=self.create_op_node("ReduceSum"),
                             args=[],
                             keywords=[ast.keyword(arg='keep_dims',
                                                   value=ast.NameConstant(
                                                       value=True))])
        call.args = [ast.BinOp(left=node.args[0],
                               op=ast.Mult(),
                               right=node.args[1]),
                     ast.UnaryOp(op=ast.USub(), operand=ast.Num(n=1))]
        call.keywords = []
        return call

    def transform_get_mask_func(self,
                                node: ast.AST,
                                enclosing_block: ast.AST,
                                insert_stmt_cb,
                                gather_idx) -> ast.Call:
        """
        Transform node_mask/edge_mask function.

        Origin code:
            g.node_mask()
        Code after transformation:
            GATHER(graph_mask, ver_subgraph_idx, 0)

        Args:
            node (ast.AST): The origin node.
            enclosing_block (ast.AST): The enclosing block for the node.
            insert_stmt_cb (Function): Insert statement callback.
            gather_idx (str): The gather idx name, can be ver_subgraph_idx or
                edge_subgraph_idx.

        Returns:
            ast.AST, call node after transformation.
        """
        return self.invoke_gather_op(ast.Name(id=GRAPH_MASK,
                                              ctx=ast.Store()), gather_idx, 0)

    def transform_scatter_idx_func(self,
                                   node: ast.AST,
                                   enclosing_block: ast.AST,
                                   insert_stmt_cb,
                                   shape_name,
                                   scatter_idx) -> ast.Call:
        """
        Transform in_degree/out_degree, num_of_nodes/num_of_edges.

        Origin code:
            g.in_degree()
        Code after transformation:
            SCATTER_ADD(
                ZEROS((n_nodes, 1), ms.int32),
                scatter_dst_idx,
                ms.ops.ones_like(scatter_dst_idx)
            )

        Args:
            shape_name (str): Shape name, in n_nodes, n_graphs.
            scatter_idx (str): The scatter idx name,
                in scatter_ver_subgraph_idx, scatter_edge_subgraph_idx,
                scatter_src_idx and scatter_dst_idx.

        Returns:
            ast.AST, call node after transformation.
        """
        call = self.invoke_op(SCATTER_ADD_OP, [
            self.create_zero_tensor(
                ast.Tuple(elts=[ast.Name(id=shape_name,
                                         ctx=ast.Load()),
                                ast.Num(n=1)],
                          ctx=ast.Load()), scatter_idx),
            ast.Name(id=scatter_idx, ctx=ast.Load()),
            ast.Call(func=self.create_op_node("ones_like"),
                     args=[ast.Name(id=scatter_idx, ctx=ast.Load())],
                     keywords=[])
        ]
                              )
        return call

    def transform_adj_to_dense_func(self,
                                    node: ast.AST,
                                    enclosing_block: ast.AST,
                                    insert_stmt_cb) -> ast.Call:
        """
        Transform adj_to_dense function.

        Origin code:
            g.adj_to_dense()
        Code after transformation:
            ms.ops.ScatterNd()(
                ms.ops.Transpose()(
                    ms.ops.Stack()([src_idx, dst_idx]),
                    (1, 0)
                ),
                ops.ones_like(src_idx),
                (n_nodes, n_nodes)
            )

        Args:

        Returns:
            ast.AST, call node after transformation.
        """
        call = ast.Call(func=ast.Call(func=self.create_op_node("ScatterNd"),
                                      args=[], keywords=[]), args=[
                                          ast.Call(func=ast.Call(func=self.create_op_node("Transpose"),
                                   args=[], keywords=[]), args=[
                                       ast.Call(func=ast.Call(func=self.create_op_node("Stack"),
                                       args=[], keywords=[]), args=[
                                           ast.List(elts=[
                        ast.Name(id=SRC_IDX, ctx=ast.Load()),
                        ast.Name(id=DST_IDX, ctx=ast.Load()),
                    ], ctx=ast.Load()),
                ], keywords=[]),
                                       ast.Tuple(elts=[
                    ast.Num(n=1),
                    ast.Num(n=0),
                ], ctx=ast.Load()),
            ], keywords=[]),
                                          ast.Call(func=self.create_op_node("ones_like"), args=[
                ast.Name(id=SRC_IDX, ctx=ast.Load()),
            ], keywords=[]),
                                              ast.Tuple(elts=[
                ast.Name(id=N_NODES, ctx=ast.Load()),
                ast.Name(id=N_NODES, ctx=ast.Load()),
            ], ctx=ast.Load())
        ], keywords=[])
        return call

    def transform_readout_func(self,
                               node: ast.AST,
                               enclosing_block: ast.AST,
                               insert_stmt_cb,
                               op_name,
                               scatter_idx,
                               is_avg):
        """
        Transform readout functions.

        Origin code:
            ret = g.sum_nodes(x)
        Code after transformation:
            SCATTER_INPUT_SNAPSHOT0 = x
            ret = SCATTER_ADD(
                ZEROS((n_graphs,) + SHAPE(SCATTER_INPUT_SNAPSHOT0)[1:],
                       ms.float32),
                scatter_ver_subgraph_idx,
                SCATTER_INPUT_SNAPSHOT0
            )

        Args:
            node (ast.AST): The origin node.
            enclosing_block (ast.AST): The enclosing block for the node.
            insert_stmt_cb (Function): Insert statement callback.
            op_name (str): Scatter op name, in SCATTER_ADD,
                SCATTER_MAX and SCATTER_MIN.
            scatter_idx (str): Scatter idx, in scatter_ver_subgraph_idx
                and scatter_edge_subgraph_idx.
            is_avg (bool): Whether it is a avg operation.

        Returns:
            ast.AST, call node after transformation.
        """
        call = self.invoke_scatter_op(node.args[0], enclosing_block,
                                      insert_stmt_cb, op_name, N_GRAPHS,
                                      scatter_idx, is_avg)
        return call

    def transform_readout_softmax_func(self, node: ast.AST,
                                       enclosing_block: ast.AST,
                                       insert_stmt_cb, scatter_idx,
                                       gather_idx):
        """
        Transform readout softmax functions.

        Origin code:
            ret = g.softmax_nodes(x)
        Code after transformation:
            EX_INPUT_SNAPSHOT0 = ms.ops.Exp()(x)
            SCATTER_INPUT_SNAPSHOT1 = EX_INPUT_SNAPSHOT0
            ret = EX_INPUT_SNAPSHOT0 / GATHER(
                SCATTER_ADD(
                    ZEROS((n_graphs,) + SHAPE(SCATTER_INPUT_SNAPSHOT1)[1:],
                    ms.float32),
                    scatter_ver_subgraph_idx,
                    SCATTER_INPUT_SNAPSHOT1
                ),
                ver_subgraph_idx,
                0
            )

        Args:
            node (ast.AST): The origin node.
            enclosing_block (ast.AST): The enclosing block for the node.
            insert_stmt_cb (Function): Insert statement callback.
            gather_idx (str): The gather idx name, can be ver_subgraph_idx
                or edge_subgraph_idx.
            scatter_idx (str): Scatter idx, in scatter_ver_subgraph_idx
                and scatter_edge_subgraph_idx.

        Returns:
            ast.AST, call node after transformation.
        """
        ex_tmp_name = "EX_INPUT_SNAPSHOT" + str(self.get_next_snapshot_id())
        tmp = ast.Assign()
        tmp.targets = [ast.Name(id=ex_tmp_name, ctx=ast.Store())]
        tmp.value = ast.Call(func=ast.Call(func=self.create_op_node("Exp"),
                                           args=[], keywords=[]),
                             args=[node.args[0]],
                             keywords=[])
        call = ast.BinOp(
            left=ast.Name(id=ex_tmp_name, ctx=ast.Store),
            op=ast.Div(),
            right=self.invoke_gather_op(
                self.invoke_scatter_op(ast.Name(id=ex_tmp_name,
                                                ctx=ast.Store()),
                                       enclosing_block, insert_stmt_cb,
                                       SCATTER_ADD_OP, N_GRAPHS, scatter_idx,
                                       False),
                gather_idx, 0
            )
        )
        insert_stmt_cb(enclosing_block, tmp, call)
        return call

    def transform_readout_broadcast_func(self,
                                         node: ast.AST,
                                         enclosing_block: ast.AST,
                                         insert_stmt_cb,
                                         gather_idx):
        """
        Transform readout broadcast functions.

        Origin code:
            ret = g.broadcast_nodes(x)
        Code after transformation:
            ret = GATHER(x, ver_subgraph_idx, 0)

        Args:
            node (ast.AST): The origin node.
            enclosing_block (ast.AST): The enclosing block for the node.
            insert_stmt_cb (Function): Insert statement callback.
            gather_idx (str): The gather idx name, can be ver_subgraph_idx
                or edge_subgraph_idx.

        Returns:
            ast.AST, call node after transformation.
        """
        return self.invoke_gather_op(node.args[0], gather_idx, 0)

    def transform_topk_func(self, node: ast.AST,
                            enclosing_block: ast.AST,
                            insert_stmt_cb):
        """
        Transform readout topk functions.

        The output code is determined by your input parameters.
        The origin code:
            ret = g.topk_nodes(x, k)
        It will be transform to:
            _, FEAT_SHAPE = SHAPE(x)
            TOPK_OUTPUT, TOPK_INDICES = ms.ops.Sort(
                                        -2, True)(X)
            ret = TOPK_OUTPUT[:, :k], TOPK_INDICES[:, :k]

        The origin code:
            ret = g.topk_nodes(x, k, sortby)
        It will be transform to:
            _, FEAT_SHAPE = SHAPE(x)
            TOPK_INDICES = ms.ops.Sort(
                -2, True)(X)[1][(..., sortby)]
            TOPK_OUTPUT = ms.ops.GatherD()(
                X,
                0,
                ms.ops.BroadcastTo((n_nodes, FEAT_SHAPE))(
                    RESHAPE(TOPK_INDICES, (n_nodes, 1))
                )
            )
            ret = TOPK_OUTPUT[:, :k], TOPK_INDICES[:, :k]

        Args:
            node (ast.AST): The origin node.
            enclosing_block (ast.AST): The enclosing block for the node.
            insert_stmt_cb (Function): Insert statement callback.


        Returns:
            ast.AST, call node after transformation.

        Raises:
            SyntaxError: be raised if input args not in (2, 3).
        """
        node_shape = "NODE_SHAPE"
        feat_shape = "FEAT_SHAPE"
        topk_output, topk_indices = "TOPK_OUTPUT", "TOPK_INDICES"
        k = node.args[1]
        x = node.args[0]
        if isinstance(k, ast.NameConstant):
            if not isinstance(k.value, int) or isinstance(k.value, bool):
                raise TypeError(f"topk function 'k' argument"
                                f"accept an int type, but got {type(k.value)}")

        call = ast.Tuple(elts=[
            ast.Subscript(value=ast.Name(id=topk_output, ctx=ast.Load()),
                          slice=ast.Slice(lower=None,
                                          upper=k,
                                          step=None),
                          ctx=ast.Load()),
            ast.Subscript(value=ast.Name(id=topk_indices, ctx=ast.Load()),
                          slice=ast.Slice(lower=None,
                                          upper=k,
                                          step=None),
                          ctx=ast.Load()),
        ], ctx=ast.Load())

        if len(node.args) == 2:
            tmp2 = ast.Assign(targets=[
                ast.Tuple(elts=[
                    ast.Name(id=topk_output, ctx=ast.Store()),
                    ast.Name(id=topk_indices, ctx=ast.Store()),
                ], ctx=ast.Store()),
            ], value=ast.Call(func=ast.Call(func=self.create_op_node("Sort"),
                                            args=[ast.UnaryOp(op=ast.USub(),
                                                              operand=ast.Num(n=2)),
                                                  ast.Constant(value=True),
                                                  ], keywords=[]),
                              args=[ast.Name(id=x.id, ctx=ast.Load())],
                              keywords=[]
                              ))
            insert_stmt_cb(enclosing_block, tmp2, call)

        elif len(node.args) == 3:
            tmp3 = ast.Assign(targets=[
                ast.Name(id=topk_output, ctx=ast.Store()),
            ], value=ast.Call(func=ast.Call(func=self.create_op_node(
                "GatherD"),
                args=[], keywords=[]), args=[
                x,
                ast.Num(n=0),
                ast.Call(func=ast.Call(func=self.create_op_node("BroadcastTo"),
                                       args=[
                                           ast.Tuple(elts=[
                                               ast.Name(id=node_shape, ctx=ast.Load()),
                                               ast.Name(id=feat_shape, ctx=ast.Load()),
                                           ], ctx=ast.Load()),
                                       ], keywords=[]), args=[
                    self.invoke_op(RESHAPE_OP, args=[
                        ast.Name(id=topk_indices, ctx=ast.Load()),
                        ast.Tuple(elts=[
                            ast.Name(id=node_shape, ctx=ast.Load()),
                            ast.Num(n=1),
                        ], keywords=[]),
                    ])
                ], keywords=[]),
            ], keywords=[])
            )
            insert_stmt_cb(enclosing_block, tmp3, call)

            sortby = node.args[2]
            if isinstance(sortby, ast.NameConstant):
                if not isinstance(sortby.value, int) or isinstance(sortby.value, bool):
                    raise TypeError(f"topk function 'sortby' argument"
                                    f"accept an int type, but got {type(sortby.value)}")

            tmp2 = ast.Assign(targets=[
                ast.Name(id=topk_indices, ctx=ast.Store()),
            ], value=ast.Subscript(value=ast.Subscript(
                value=ast.Call(func=ast.Call(func=self.create_op_node("Sort"),
                                             args=[
                                                 ast.UnaryOp(op=ast.USub(), operand=ast.Num(n=2)),
                                                 ast.Constant(value=True),
                                             ], keywords=[]), args=[x],
                               keywords=[]),
                slice=ast.Index(value=ast.Num(n=1)), ctx=ast.Load()),
                slice=ast.Index(value=ast.Tuple(
                    elts=[
                        ast.Ellipsis(),
                        sortby,
                    ],
                    ctx=ast.Load())),
                ctx=ast.Load())
            )
            insert_stmt_cb(enclosing_block, tmp2, call)

        else:
            raise SyntaxError("Topk function only accept 2 or 3 args.")

        tmp0 = ast.Assign(targets=[
            ast.Tuple(elts=[
                ast.Name(id=node_shape, ctx=ast.Store()),
                ast.Name(id=feat_shape, ctx=ast.Store()),
            ], ctx=ast.Store()),
        ], value=self.invoke_op(SHAPE_OP, args=[x]))
        insert_stmt_cb(enclosing_block, tmp0, call)

        return call

    def transform_readout_topk_func(self, node: ast.AST,
                                    enclosing_block: ast.AST,
                                    insert_stmt_cb, gather_idx):
        """
        Transform readout topk functions.

        The output code is determined by your input parameters.
        The origin code:
            ret = g.topk_nodes(x, k)
        It will be transform to:
            _, FEAT_SHAPE = SHAPE(x)
            X_SEPRATED_BY_GRAPH = SCATTER_ADD(
                ZEROS((n_graphs, n_nodes, FEAT_SHAPE), ms.float32),
                ms.ops.Transpose()(
                    ms.ops.Stack()([ver_subgraph_idx,
                                    ms.nn.Range(0, n_nodes, 1)()]),
                    (1, 0)
                ),
                x
            )
            TOPK_OUTPUT, TOPK_INDICES = ms.ops.Sort(
                                        -2, True)(X_SEPRATED_BY_GRAPH)
            ret = TOPK_OUTPUT[:, :k], TOPK_INDICES[:, :k]

        The origin code:
            ret = g.topk_nodes(x, k, sortby)
        It will be transform to:
            _, FEAT_SHAPE = SHAPE(x)
            X_SEPRATED_BY_GRAPH = SCATTER_ADD(
                ZEROS((n_graphs, n_nodes, FEAT_SHAPE), ms.float32),
                ms.ops.Transpose()(
                    ms.ops.Stack()([ver_subgraph_idx,
                                    ms.nn.Range(0, n_nodes, 1)()]),
                    (1, 0)
                ),
                x
            )
            TOPK_INDICES = ms.ops.Sort(
                -2, True)(X_SEPRATED_BY_GRAPH)[1][(..., sortby)]
            TOPK_OUTPUT = ms.ops.GatherD()(
                X_SEPRATED_BY_GRAPH,
                1,
                ms.ops.BroadcastTo((n_graphs, n_nodes, FEAT_SHAPE))(
                    RESHAPE(TOPK_INDICES, (n_graphs, n_nodes, 1))
                )
            )
            ret = TOPK_OUTPUT[:, :k], TOPK_INDICES[:, :k]

        Args:
            node (ast.AST): The origin node.
            enclosing_block (ast.AST): The enclosing block for the node.
            insert_stmt_cb (Function): Insert statement callback.
            gather_idx (str): The gather idx name, can be ver_subgraph_idx
                or edge_subgraph_idx.

        Returns:
            ast.AST, call node after transformation.

        Raises:
            SyntaxError: be raised if input args not in (2, 3).
        """
        node_shape = "NODE_SHAPE"
        feat_shape = "FEAT_SHAPE"
        x_seperated = "X_SEPRATED_BY_GRAPH"
        inf_ = "_INF_FILL_"
        fill_ = "FILL_"
        topk_output, topk_indices = "TOPK_OUTPUT", "TOPK_INDICES"
        k = node.args[1]
        x = node.args[0]
        if isinstance(k, ast.NameConstant):
            if not isinstance(k.value, int) or isinstance(k.value, bool):
                raise TypeError(f"topk function 'k' argument"
                                f"accept an int type, but got {type(k.value)}")

        call = ast.Tuple(elts=[
            ast.Subscript(value=ast.Name(id=topk_output, ctx=ast.Load()),
                          slice=ast.ExtSlice(dims=[ast.Slice(lower=None,
                                                             upper=None,
                                                             step=None),
                                                   ast.Slice(lower=None,
                                                             upper=k,
                                                             step=None),
                                                   ]), ctx=ast.Load()),
            ast.Subscript(value=ast.Name(id=topk_indices, ctx=ast.Load()),
                          slice=ast.ExtSlice(dims=[ast.Slice(lower=None,
                                                             upper=None,
                                                             step=None),
                                                   ast.Slice(lower=None,
                                                             upper=k,
                                                             step=None),
                                                   ]), ctx=ast.Load()),
        ], ctx=ast.Load())

        if len(node.args) == 2:
            tmp4 = ast.Assign(targets=[
                ast.Tuple(elts=[
                    ast.Name(id=topk_output, ctx=ast.Store()),
                    ast.Name(id=topk_indices, ctx=ast.Store()),
                ], ctx=ast.Store()),
            ], value=ast.Call(func=ast.Call(func=self.create_op_node("Sort"),
                                            args=[ast.UnaryOp(op=ast.USub(),
                                                              operand=ast.Num(n=2)),
                                                  ast.Constant(value=True),
                                                  ], keywords=[]),
                              args=[ast.Name(id=x_seperated, ctx=ast.Load())],
                              keywords=[]
                              ))
            insert_stmt_cb(enclosing_block, tmp4, call)

        elif len(node.args) == 3:
            tmp5 = ast.Assign(targets=[
                ast.Name(id=topk_output, ctx=ast.Store()),
            ], value=ast.Call(func=ast.Call(func=self.create_op_node(
                "GatherD"),
                args=[], keywords=[]), args=[
                ast.Name(id=x_seperated, ctx=ast.Load()),
                ast.Num(n=1),
                ast.Call(func=ast.Call(func=self.create_op_node("BroadcastTo"),
                                       args=[
                                           ast.Tuple(elts=[
                                               ast.Name(id=N_GRAPHS, ctx=ast.Load()),
                                               ast.Name(id=node_shape, ctx=ast.Load()),
                                               ast.Name(id=feat_shape, ctx=ast.Load()),
                                           ], ctx=ast.Load()),
                                       ], keywords=[]), args=[
                    self.invoke_op(RESHAPE_OP, args=[
                        ast.Name(id=topk_indices, ctx=ast.Load()),
                        ast.Tuple(elts=[
                            ast.Name(id=N_GRAPHS, ctx=ast.Load()),
                            ast.Name(id=node_shape, ctx=ast.Load()),
                            ast.Num(n=1),
                        ], keywords=[]),
                    ])
                ], keywords=[]),
            ], keywords=[])
            )
            insert_stmt_cb(enclosing_block, tmp5, call)

            sortby = node.args[2]
            if isinstance(sortby, ast.NameConstant):
                if not isinstance(sortby.value, int) or isinstance(sortby.value, bool):
                    raise TypeError(f"topk function 'sortby' argument"
                                    f"accept an int type, but got {type(sortby.value)}")

            tmp4 = ast.Assign(targets=[
                ast.Name(id=topk_indices, ctx=ast.Store()),
            ],
                value=ast.Subscript(value=ast.Subscript(
                    value=ast.Call(func=ast.Call(func=self.create_op_node("Sort"),
                                                 args=[
                                                     ast.UnaryOp(op=ast.USub(), operand=ast.Num(n=2)),
                                                     ast.Constant(value=True)],
                                                 keywords=[]),
                                   args=[ast.Name(id=x_seperated,
                                                  ctx=ast.Load())],
                                   keywords=[]),
                    slice=ast.Index(value=ast.Num(n=1)), ctx=ast.Load()),
                    slice=ast.Index(value=ast.Tuple(
                        elts=[
                            ast.Ellipsis(),
                            sortby,
                        ],
                        ctx=ast.Load())),
                    ctx=ast.Load())
            )
            insert_stmt_cb(enclosing_block, tmp4, call)

        else:
            raise SyntaxError("Topk function only accept 2 or 3 args.")
        long_args = [
            ast.List(elts=[
                ast.Name(id=gather_idx,
                         ctx=ast.Load()),
                ast.Call(
                    func=ast.Call(
                        func=self.create_op_node("Range", "nn"),
                        args=[
                            ast.Num(n=0),
                            ast.Name(id=node_shape,
                                     ctx=ast.Load()),
                            ast.Num(n=1)],
                        keywords=[]), args=[], keywords=[]),
            ], ctx=ast.Load()),
        ]

        tmp3 = ast.Assign(targets=[ast.Name(id=x_seperated, ctx=ast.Store())],
                          value=ast.Call(func=ast.Call(self.create_op_node("TensorScatterUpdate"),
                                                       args=[], keywords=[]),
                                         args=[ast.Name(id=inf_, ctx=ast.Load()),
                                               ast.Call(func=ast.Call(func=self.create_op_node("Transpose"),
                                                                      args=[], keywords=[]),
                                                        args=[ast.Call(func=ast.Call(func=self.create_op_node("Stack"),
                                                                                     args=[], keywords=[]),
                                                                       args=long_args,
                                                                       keywords=[]),
                                                              ast.Tuple(elts=[ast.Num(n=1), ast.Num(n=0)],
                                                                        ctx=ast.Load()),
                                                              ],
                                                        keywords=[]),
                                               x],
                                         keywords=[]
                                         )
                          )
        insert_stmt_cb(enclosing_block, tmp3, call)

        tmp2 = ast.Assign(targets=[ast.Name(id=inf_, ctx=ast.Store())],
                          value=ast.Call(func=self.create_op_node('mul'),
                                         args=[
                                             ast.Call(func=ast.Name(FILL_OP, ctx=ast.Load()),
                                                      args=[ast.Attribute(value=x,
                                                                          attr='dtype',
                                                                          ctx=ast.Load()),
                                                            ast.Tuple(elts=[ast.Name(id=N_GRAPHS, ctx=ast.Load()),
                                                                            ast.Name(id=node_shape, ctx=ast.Load()),
                                                                            ast.Name(id=feat_shape, ctx=ast.Load()),
                                                                            ], ctx=ast.Load()),
                                                            ast.Num(n=1.)],
                                                      keywords=[]),
                                             ast.Name(id=fill_, ctx=ast.Load())
                                         ], keywords=[]
                                         )
                          )
        insert_stmt_cb(enclosing_block, tmp2, call)

        tmp1 = ast.Assign(targets=[ast.Name(id=fill_, ctx=ast.Store())],
                          value=ast.Call(func=ast.Call(func=self.create_op_node("Sub"),
                                                       args=[], keywords=[]),
                                         args=[ast.Call(func=ast.Call(func=self.create_op_node("ReduceMin"),
                                                                      args=[], keywords=[]),
                                                        args=[x],
                                                        keywords=[]),
                                               ast.Call(func=ast.Attribute(value=ast.Name(id=backend()),
                                                                           attr="Tensor", ctx=ast.Load()),
                                                        args=[ast.Num(n=0.0001)],
                                                        keywords=[])],
                                         keywords=[]))
        insert_stmt_cb(enclosing_block, tmp1, call)

        tmp0 = ast.Assign(targets=[
            ast.Tuple(elts=[
                ast.Name(id=node_shape, ctx=ast.Store()),
                ast.Name(id=feat_shape, ctx=ast.Store()),
            ], ctx=ast.Store()),
        ], value=self.invoke_op(SHAPE_OP, args=[x]))
        insert_stmt_cb(enclosing_block, tmp0, call)

        return call

    def transform_get_homo_func(self,
                                node: ast.AST,
                                enclosing_block: ast.AST,
                                insert_stmt_cb):
        if len(node.args) != 1:
            raise SyntaxError(
                "Get homo graph function should have only one args.")
        return ast.List(elts=[
            ast.Subscript(value=ast.Name(id=field_name, ctx=ast.Load()),
                          slice=ast.Index(value=node.args[0]), ctx=ast.Load())
            for field_name in GRAPH_FIELD_NAMES
        ], ctx=ast.Load())

    def init_op(self,
                alias: str,
                op_name: str,
                op_args: List[ast.AST] = None,
                op_keywords: List[ast.AST] = None):
        """helper function, init op."""
        if op_args is None:
            op_args = []
        if op_keywords is None:
            op_keywords = []
        assign = ast.Assign()
        assign.targets = [ast.Name(id=alias, ctx=ast.Store())]
        assign.value = ast.Call(func=self.create_op_node(op_name),
                                args=op_args,
                                keywords=op_keywords)
        return assign

    def invoke_op(self,
                  alias: str,
                  args: List[ast.AST] = None,
                  keywords: List[ast.AST] = None):
        """helper function, invoke op."""
        if args is None:
            args = []
        if keywords is None:
            keywords = []
        return ast.Call(func=ast.Name(id=alias, ctx=ast.Load()),
                        args=args, keywords=keywords)

    def invoke_gather_op(self, node: ast.AST, idx, value):
        """helper function, invoke gather op."""
        return self.invoke_op(
            GATHER_OP, [node,
                        ast.Name(id=idx, ctx=ast.Store()),
                        ast.Constant(value=value)])

    def invoke_masked_fill_op(self, node: ast.AST, value):
        """helper function, invoke masked_fill op."""
        return self.invoke_op(MASKED_FILL_OP,
                              [node,
                               self.invoke_op(IS_INF_OP, [node]),
                               ast.Constant(value=value)])

    def invoke_scatter_op(self,
                          feat,
                          enclosing_block,
                          insert_stmt_cb,
                          scatter_op,
                          shape_name,
                          dst_idx,
                          is_avg):
        """helper function, invoke scatter op."""
        new_func = ast.Name(id=scatter_op, ctx=ast.Load())
        call = ast.Call()
        call.func = new_func

        scatter_tmp_name = "SCATTER_INPUT_SNAPSHOT" + \
            str(self.get_next_snapshot_id())
        tmp = ast.Assign()
        tmp.targets = [ast.Name(id=scatter_tmp_name, ctx=ast.Store())]
        tmp.value = feat
        output_shape_node = ast.BinOp(
            left=ast.Tuple(
                elts=[ast.Name(id=shape_name, ctx=ast.Load())],
                ctx=ast.Load()),
            op=ast.Add(),
            right=ast.Subscript(
                value=self.invoke_op(
                    SHAPE_OP,
                    args=[ast.Name(id=scatter_tmp_name, ctx=ast.Load())]),
                slice=ast.Slice(lower=ast.Constant(value=1), upper=None,
                                step=None), ctx=ast.Load()))
        if scatter_op in [SCATTER_MAX_OP, SCATTER_MIN_OP]:
            input_x = self.create_inf_tensor(scatter_op, output_shape_node, scatter_tmp_name)
        else:
            input_x = self.create_zero_tensor(output_shape_node, scatter_tmp_name)
        call.args = [input_x,
                        ast.Name(id=dst_idx, ctx=ast.Load()),
                        ast.Name(id=scatter_tmp_name, ctx=ast.Load())]
        call.keywords = []
        if is_avg:
            div_right_node = ast.BinOp(
                left=self.transform_scatter_idx_func(feat, enclosing_block, insert_stmt_cb, shape_name, dst_idx),
                op=ast.Add(),
                right=ast.Constant(value=1e-15)
            )
            call = ast.BinOp(left=call,
                                op=ast.Div(),
                                right=div_right_node)
        elif scatter_op in [SCATTER_MAX_OP, SCATTER_MIN_OP]:
            call = self.invoke_masked_fill_op(call, 0.0)
        insert_stmt_cb(enclosing_block, tmp, call)
        return call

    def invoke_csr_reduce(self,
                          feat,
                          enclosing_block,
                          insert_stmt_cb,
                          scatter_op,
                          shape_name,
                          dst_idx,
                          is_avg):
        """helper function, invoke csr reduce op."""
        new_func = ast.Name(id="RESHAPE", ctx=ast.Load())
        call = ast.Call()
        call.func = new_func

        tmp_res = "SCATTER_INPUT_SNAPSHOT" + \
                    str(self.get_next_snapshot_id())
        tmp = ast.Assign()
        tmp.targets = [ast.Name(id=tmp_res, ctx=ast.Store())]
        tmp.value = feat

        tmp_param1 = ast.BinOp(
            left=ast.Tuple(
                elts=[ast.Name(id=N_NODES, ctx=ast.Load()),
                        ast.Name(id=N_NODES, ctx=ast.Load())],
                ctx=ast.Load()),
            op=ast.Add(),
            right=ast.Subscript(
                value=self.invoke_op(
                    SHAPE_OP,
                    args=[ast.Name(id=tmp_res, ctx=ast.Load())]),
                slice=ast.Slice(lower=ast.Constant(value=1), upper=None,
                                step=None), ctx=ast.Load()))

        tmp2_res = "SCATTER_INPUT_SNAPSHOT" + \
                    str(self.get_next_snapshot_id())
        tmp2 = ast.Assign()
        tmp2.targets = [ast.Name(id=tmp2_res, ctx=ast.Store())]
        if self.backward:
            tmp2_op_name = 'self.CSR_BACKWARD_REDUCE_SUM'
            tmp2.value = ast.Call(func=ast.Name(id=tmp2_op_name),
                                    args=[ast.Name(INDPTR),
                                        ast.Name(INDICES),
                                        ast.Name(tmp_res),
                                        tmp_param1,
                                        ast.Constant(1),
                                        ast.Name(INDICES_BACKWARD)
                                        ],
                                    keywords=[])
        else:
            tmp2_op_name = 'CSR_REDUCE_SUM'
            tmp2.value = ast.Call(func=ast.Name(id=tmp2_op_name),
                                    args=[ast.Name(INDPTR),
                                        ast.Name(INDICES),
                                        ast.Name(tmp_res),
                                        tmp_param1,
                                        ast.Constant(1)
                                        ],
                                    keywords=[])

        insert_stmt_cb(enclosing_block, tmp2, call)
        insert_stmt_cb(enclosing_block, tmp, call)

        tmp_param2 = ast.BinOp(
            left=ast.Tuple(
                elts=[ast.Name(id=N_NODES, ctx=ast.Load())],
                ctx=ast.Load()),
            op=ast.Add(),
            right=ast.Subscript(
                value=self.invoke_op(
                    SHAPE_OP,
                    args=[ast.Name(id=tmp2_res)]),
                slice=ast.Slice(lower=ast.Constant(value=2), upper=None,
                                step=None), ctx=ast.Load()))
        call.args = [ast.Name(tmp2_res), tmp_param2]
        call.keywords = []
        return call


    def invoke_subscript_index(self, node, index):
        """helper function, invoke subscript index."""
        return ast.Subscript(value=node,
                             slice=ast.Index(value=ast.Constant(value=index)),
                             ctx=ast.Load())

    def invoke_init_intermediate(self, target_name, value_name):
        """helper function, invoke init intermediate."""
        assign = ast.Assign()
        assign.targets = [ast.Name(id=target_name, ctx=ast.Store())]
        assign.value = self.invoke_op(
            RESHAPE_OP,
            args=[ast.Name(value_name, ctx=ast.Load()),
                  ast.Tuple(elts=[self.invoke_subscript_index(
                      self.invoke_op(
                          SHAPE_OP,
                          args=[ast.Name(
                              value_name, ctx=ast.Load())]),
                      index=0),
                      ast.Constant(value=1)],
                            ctx=ast.Load())])
        return assign

    def invoke_get_shape(self, shape_name, value_name):
        """helper function, invoke get shape."""
        assign = ast.Assign()
        assign.targets = [ast.Name(id=shape_name, ctx=ast.Load())]
        assign.value = self.invoke_subscript_index(
            self.invoke_op(
                SHAPE_OP,
                [ast.Name(id=value_name, ctx=ast.Load())]), index=0)
        return assign

    def create_op_node(self, op_name: str, pack="ops"):
        """helper function, create a op node."""
        if backend() is None:
            raise TypeError("Backend name is unknown. Please set_backend first.")
        return ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(
                    id=backend(),
                    ctx=ast.Load()),
                attr=pack, ctx=ast.Load()), attr=op_name,
            ctx=ast.Load())

    def create_zero_tensor(self, shape_ast: ast.AST, dtype_id: str):
        """helper function, create zero tensor."""
        if backend() is None:
            raise TypeError("Backend name is unknown. Please set_backend first.")
        if dtype_id is None:
            dtype_id = backend()
            attr_dtype = 'float32'
        else:
            attr_dtype = 'dtype'
        return ast.Call(
            func=ast.Name(ZEROS_OP, ctx=ast.Load()),
            args=[shape_ast,
                  ast.Attribute(
                      value=ast.Name(
                          id=dtype_id,
                          ctx=ast.Load()),
                      attr=attr_dtype)],
            keywords=[])

    def create_inf_tensor(self, scatter_op: str, shape_ast: ast.AST, dtype_id: str):
        """helper function, create inf tensor."""
        if backend() is None:
            raise TypeError("Backend name is unknown. Please set_backend first.")
        if dtype_id is None:
            dtype_id = backend()
            attr_dtype = 'float32'
        else:
            attr_dtype = 'dtype'
        const_val = self.neg_inf if scatter_op == SCATTER_MAX_OP else self.pos_inf
        return ast.Call(
            func=ast.Name(FILL_OP, ctx=ast.Load()),
            args=[ast.Attribute(
                      value=ast.Name(
                          id=dtype_id,
                          ctx=ast.Load()),
                      attr=attr_dtype),
                shape_ast,
                ast.Constant(value=const_val)],
            keywords=[])
