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
"""Ast rewriter."""
import ast
from types import MethodType
from typing import List

from .ast_base import BaseAstTransformer
from .ast_transformer import InsertStatementBeforeCurrent


class ProgBlockCtx:
    """Program block context class."""

    def __init__(self, linear_prog_ctx):
        self.prog_ctx_ = linear_prog_ctx

    def __enter__(self):
        return self.prog_ctx_[-1]

    def __exit__(self, exc_type, exc_value, tb):
        self.prog_ctx_.pop()


class AstRewriter(BaseAstTransformer):
    """Ast rewriter."""

    def __init__(self, transformation_dict: map):
        super().__init__()
        self.prog_blocks = [[]]
        self.trans_dict_ = transformation_dict
        setattr(self.__class__, "visit_FunctionDef", self.visit_functiondef)
        setattr(self.__class__, "visit_If", self.visit_if)
        setattr(self.__class__, "visit_For", self.visit_for)
        setattr(self.__class__, "visit_While", self.visit_while)

        def rewrite_fn(fn):
            """
            Rewrite function.

            Args:
                fn (Function): the function to rewrite.
            """

            def rewrite_fn_impl(self_obj, node: ast.AST):
                """
                Rewrite function implementation.

                Args:
                    node (ast.AST): the node to rewrite.
                    self_obj: self

                Returns:
                    ast.AST, node after rewrite.
                """
                node = fn(node)
                if node in self_obj.trans_dict_:
                    trans_sequence = self_obj.trans_dict_[node]
                    for trans in trans_sequence:
                        node = trans.transform(
                            node,
                            self_obj.cur_block(),
                            self_obj.insert_stmt_before_cur_cb)

                return node

            return rewrite_fn_impl

        for k in self.supported_constructs_:
            fn_name = "visit_" + k
            original_fn = self.default_fn \
                if not hasattr(self, fn_name) \
                else getattr(self, fn_name)
            setattr(self, fn_name, MethodType(rewrite_fn(original_fn), self))

    def visit_functiondef(self, node: ast.FunctionDef):
        """Visit ast.FunctionDef."""
        with self.new_block(node):
            self.generic_visit(node)
        return node

    def visit_if(self, node: ast.If):
        """Visit ast.If."""
        with self.new_block(node):
            self.generic_visit(node)
        return node

    def visit_for(self, node: ast.For):
        """Visit ast.For."""
        with self.new_block(node):
            self.generic_visit(node)
        return node

    def visit_while(self, node: ast.While):
        """Visit ast.While."""
        with self.new_block(node):
            self.generic_visit(node)
        return node

    def default_fn(self, node: ast.AST):
        """Default visit function."""
        self.generic_visit(node)
        return node

    def new_block(self, new_block: List[ast.AST]):
        """Create a new block."""
        self.prog_blocks.append(new_block)
        return ProgBlockCtx(self.prog_blocks)

    def insert_stmt_before_cur_cb(self,
                                  node: ast.AST,
                                  new_stmt: ast.AST,
                                  cur_node: ast.AST):
        """
        Call back for insert a statement before current node.

        Args:
            node (ast.AST): the function node.
            new_stmt (ast.AST): the new statement to insert.
            cur_node (ast.AST): current node.

        Returns:
            ast.AST, the function node after insert the new statement.
        """
        transformation = InsertStatementBeforeCurrent(new_stmt, cur_node)
        if node in self.trans_dict_:
            self.trans_dict_[node].insert(0, transformation)
        else:
            self.trans_dict_[node] = [transformation]

    def cur_block(self):
        """Get the current block."""
        return self.prog_blocks[-1]
