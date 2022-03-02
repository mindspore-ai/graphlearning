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
"""Ast Transformer."""
import ast
from typing import List

from .vectorization import VectorizationType
from .ast_base import BaseAstTransformer, FindChild
from .code_comparator import trace_stmt_insertion, trace_stmt_replacement


class ReplaceName(BaseAstTransformer):
    """Replace Name Transformer."""

    def __init__(self,
                 old_id: str,
                 new_id: str):
        super().__init__()
        self.old_id_ = old_id
        self.new_id_ = new_id
        setattr(self.__class__, "visit_Name", self.visit_name)

    def transform(self,
                  node: ast.AST,
                  enclosing_block: ast.AST = None,
                  insert_stmt_before_cur_cb=None) -> ast.AST:
        """
        Transform function. The old id will be replaced with the new id.

        Args:
            node (ast.AST): The node to transform.
            enclosing_block (ast.AST): The enclosing block that the node
                belonging to.
            insert_stmt_before_cur_cb (ast.AST): The call back function to
                insert statement.

        Returns:
            ast.AST, node after transformation.
        """
        self.generic_visit(node)
        if enclosing_block is None:
            pass
        if insert_stmt_before_cur_cb is None:
            pass
        return node

    def visit_name(self, node: ast.Name) -> ast.Name:
        """Visit ast.Name."""
        if node.id == self.old_id_:
            setattr(node, "id", self.new_id_)
        return node


class CastGraphType(BaseAstTransformer):
    """Cast Graph Type Transformer."""

    def __init__(self,
                 old_type: VectorizationType,
                 new_type: VectorizationType):
        super().__init__()
        self.old_type_ = old_type
        self.new_type_ = new_type

    def transform(self,
                  node: ast.AST,
                  enclosing_block: ast.AST = None,
                  insert_stmt_before_cur_cb=None) -> ast.AST:
        """
        Transform function. Gather node will be transformed here.

        Args:
            node (ast.AST): The node to transform.
            enclosing_block (ast.AST): The enclosing block that the node
                belonging to.
            insert_stmt_before_cur_cb (ast.AST): The call back function to
                insert statement.

        Returns:
            ast.AST, node after transformation.
        """
        if self.new_type_ == VectorizationType.EDGE and self.old_type_:
            assert self.old_type_ in {VectorizationType.VERTEX, VectorizationType.SRC, VectorizationType.DST}
            call = self.backend_.create_gather_node(node, self.old_type_)
            ast.copy_location(call, node)
            ast.fix_missing_locations(call)
            return call
        if enclosing_block is None:
            pass
        if insert_stmt_before_cur_cb is None:
            pass
        return node


class InlineAttributeSetter(BaseAstTransformer):
    """Inline Attribute Setter Transformer."""

    def __init__(self, node_type):
        super().__init__()
        self.type = node_type

    def transform(self,
                  node: ast.AST,
                  enclosing_block: ast.AST = None,
                  insert_stmt_before_cur_cb=None) -> ast.AST:
        """
        Transform function. Set attribute function of Graph will be
        transformed here.

        Args:
            node (ast.AST): The node to transform.
            enclosing_block (ast.AST): The enclosing block that the node
                belonging to.
            insert_stmt_before_cur_cb (ast.AST): The call back function to
                insert statement.

        Returns:
            ast.AST, node after transformation.
        """
        err_str = f"Attribute setting handler only deals " \
                  f"with Expr node but received " \
                  f"{node.__class__} in Line {node.lineno}"
        assert isinstance(node, ast.Expr), err_str
        call_node = node.value
        err_str = "AttributeSetter's input's attribute \"value\" must " \
                  "be type ast.Call."
        assert isinstance(call_node, ast.Call), err_str
        assign_node = self.backend_.inline_attribute_setter(node)
        ast.copy_location(assign_node, node)
        ast.fix_missing_locations(assign_node)
        trace_stmt_replacement(node, assign_node)
        if enclosing_block is None:
            pass
        if insert_stmt_before_cur_cb is None:
            pass
        return assign_node


class InlineSupportedOpCall(BaseAstTransformer):
    """Inline SupportedOp Call Transformer."""

    def __init__(self, node_info):
        super().__init__()
        self.func_name = node_info.func_name
        self.args = node_info.args

    def transform(self,
                  node: ast.AST,
                  enclosing_block: ast.AST = None,
                  insert_stmt_before_cur_cb=None) -> ast.AST:
        """
        Transform function. SupportedOp will be transformed according to
        their requirements here.

        Args:
            node (ast.AST): The node to transform.
            enclosing_block (ast.AST): The enclosing block that the node
                belonging to.
            insert_stmt_before_cur_cb (ast.AST): The call back function to
                insert statement.

        Returns:
            ast.AST, node after transformation.
        """
        func = getattr(self.backend_, self.func_name)
        call = func(node,
                    enclosing_block,
                    insert_stmt_before_cur_cb,
                    *self.args)
        ast.copy_location(call, node)
        ast.fix_missing_locations(call)
        return call


class DefaultArgsSetter(BaseAstTransformer):
    """Default Arguments Setter Transformer."""

    def __init__(self, unused_args):
        super().__init__()
        self.unused_args = unused_args

    def transform(self,
                  node: ast.AST,
                  enclosing_block: ast.AST = None,
                  insert_stmt_before_cur_cb=None) -> ast.AST:
        """
        Transform function. Three None Constant will be set in the function
        definition here.

        Args:
            node (ast.AST): The node to transform.
            enclosing_block (ast.AST): The enclosing block that the node
                belonging to.
            insert_stmt_before_cur_cb (ast.AST): The call back function to
                insert statement.

        Returns:
            ast.AST, node after transformation.
        """
        for i in range(self.unused_args):
            node.args.args.append(ast.arg(arg="UNUSED_" + str(i),
                                          annotation=None))
            node.args.defaults.append(ast.NameConstant(value=None))
        if enclosing_block is None:
            pass
        if insert_stmt_before_cur_cb is None:
            pass
        return node


class FlattenAttribute(BaseAstTransformer):
    """Flatten Attribute Transformer."""

    def transform(self,
                  node: ast.AST,
                  enclosing_block: ast.AST = None,
                  insert_stmt_before_cur_cb=None) -> ast.AST:
        """
        Transform function. Extract the name from it's attribute. E.g., 'g.v'
        will be transformed to 'v'.

        Args:
            node (ast.AST): The node to transform.
            enclosing_block (ast.AST): The enclosing block that the node
                belonging to.
            insert_stmt_before_cur_cb (ast.AST): The call back function to
                insert statement.

        Returns:
            ast.AST, node after transformation.
        """
        new_node = ast.Name()
        new_node.id = node.attr
        ast.copy_location(new_node, node)
        ast.fix_missing_locations(new_node)
        if enclosing_block is None:
            pass
        if insert_stmt_before_cur_cb is None:
            pass
        return new_node


class VectorizeFor(BaseAstTransformer):
    """Vectorize For Transformer."""

    def transform(self,
                  node: ast.AST,
                  enclosing_block: ast.AST = None,
                  insert_stmt_before_cur_cb=None) -> ast.AST:
        """
        Transform function. For loop will be removed here.

        Args:
            node (ast.AST): The node to transform.
            enclosing_block (ast.AST): The enclosing block that the node
                belonging to.
            insert_stmt_before_cur_cb (ast.AST): The call back function to
                insert statement.

        Returns:
            ast.AST, node after transformation.
        """
        if enclosing_block is None:
            pass
        if insert_stmt_before_cur_cb is None:
            pass
        return node.body


class VectorizeListComp(BaseAstTransformer):
    """Vectorize ListComp Transformer."""

    def transform(self,
                  node: ast.AST,
                  enclosing_block: ast.AST = None,
                  insert_stmt_before_cur_cb=None) -> ast.AST:
        """
        Transform function. ListComp will be replaced by it's elt here.

        Args:
            node (ast.AST): The node to transform.
            enclosing_block (ast.AST): The enclosing block that the node
                belonging to.
            insert_stmt_before_cur_cb (ast.AST): The call back function to
                insert statement.

        Returns:
            ast.AST, node after transformation.
        """
        if not hasattr(node, "elt"):
            err_msg = f"node {node} in line {node.lineno} missing elt."
            raise SyntaxError(err_msg)
        if enclosing_block is None:
            pass
        if insert_stmt_before_cur_cb is None:
            pass
        return node.elt


class InitBackend(BaseAstTransformer):
    """Init Backend Transformer."""

    def __init__(self, graph_type):
        super().__init__()
        # graph type can be "Graph" or "BatchedGraph".
        self.graph_type = graph_type

    def transform(self,
                  node: ast.AST,
                  enclosing_block: ast.AST = None,
                  insert_stmt_before_cur_cb=None) -> ast.AST:
        """
        Transform function. Add predefined Expr at the beginning of the
        function.

        Args:
            node (ast.AST): The node to transform.
            enclosing_block (ast.AST): The enclosing block that the node
                belonging to.
            insert_stmt_before_cur_cb (ast.AST): The call back function to
                insert statement.

        Returns:
            ast.AST, node after transformation.
        """
        node.body = self.backend_.init_intermediates(self.graph_type) + \
            node.body
        node.body = self.backend_.init_graph_indices() + node.body
        node.body = self.backend_.init_ops() + node.body
        ast.fix_missing_locations(node)
        if enclosing_block is None:
            pass
        if insert_stmt_before_cur_cb is None:
            pass
        return node


class ArgReplacer(BaseAstTransformer):
    """Argument Replacer Transformer."""

    def __init__(self, new_args: List[str], replace_pos):
        super().__init__()
        self.new_args_ = new_args
        self.pos_ = replace_pos

    def transform(self,
                  node: ast.AST,
                  enclosing_block: ast.AST = None,
                  insert_stmt_before_cur_cb=None) -> ast.AST:
        """
        Transform function. Replace the Graph/BatchedGraph to the arguments.

        Args:
            node (ast.AST): The node to transform.
            enclosing_block (ast.AST): The enclosing block that the node
                belonging to.
            insert_stmt_before_cur_cb (ast.AST): The call back function to
                insert statement.

        Returns:
            ast.AST, node after transformation.
        """
        assert self.pos_ == len(node.args.args) - 1,\
            "Can only replace the last argument"
        new_args_node = [ast.arg(arg, annotation=None)
                         for arg in self.new_args_]
        setattr(node.args, 'args', node.args.args[0:self.pos_] +
                new_args_node + node.args.args[self.pos_ + 1:])
        ast.fix_missing_locations(node)
        if enclosing_block is None:
            pass
        if insert_stmt_before_cur_cb is None:
            pass
        return node


class ArgListReplacer(BaseAstTransformer):
    """Argument List Replacer Transformer."""

    def __init__(self, new_args: List[str], insert_pos: int):
        super().__init__()
        self.new_args_ = new_args
        self.pos_ = insert_pos

    def transform(self,
                  node: ast.AST,
                  enclosing_block: ast.AST = None,
                  insert_stmt_before_cur_cb=None) -> ast.AST:
        """
        Transform function. Replace the Graph/BatchedGraph to the arguments.

        Args:
            node (ast.AST): The node to transform.
            enclosing_block (ast.AST): The enclosing block that the node
                belonging to.
            insert_stmt_before_cur_cb (ast.AST): The call back function to
                insert statement.

        Returns:
            ast.AST, node after transformation.
        """
        assert self.pos_ == len(node.args) - 1,\
            "Can only replace the last argument"
        new_args_node = [ast.Name(arg) for arg in self.new_args_]
        setattr(node, 'args', node.args[0:self.pos_] + new_args_node +
                node.args[self.pos_ + 1:])
        ast.fix_missing_locations(node)
        if enclosing_block is None:
            pass
        if insert_stmt_before_cur_cb is None:
            pass
        return node


class InsertStatementBeforeCurrent(BaseAstTransformer):
    """Insert Statement Before Current Transformer."""

    def __init__(self, new_stmt: ast.AST, cur_ast: ast.AST):
        super().__init__()
        self.new_stmt_ = new_stmt
        self.cur_ast_ = cur_ast

    def transform(self,
                  node: ast.AST,
                  enclosing_block: ast.AST = None,
                  insert_stmt_before_cur_cb=None) -> ast.AST:
        """
        Transform function. Insert statement operation will occurred here.

        Args:
            node (ast.AST): The node to transform.
            enclosing_block (ast.AST): The enclosing block that the node
                belonging to.
            insert_stmt_before_cur_cb (ast.AST): The call back function to
                insert statement.

        Returns:
            ast.AST, node after transformation.
        """
        finder = FindChild()
        for field, value in ast.iter_fields(node):
            if isinstance(value, List):
                stmt_list = value
                new_stmt_list = []
                for stmt in stmt_list:
                    if finder.find_child(stmt, self.cur_ast_):
                        trace_stmt_insertion(stmt, self.new_stmt_)
                        new_stmt_list.append(self.new_stmt_)
                        new_stmt_list.append(stmt)
                    else:
                        new_stmt_list.append(stmt)
                setattr(node, field, new_stmt_list)
        if enclosing_block is None:
            pass
        if insert_stmt_before_cur_cb is None:
            pass
        return node
