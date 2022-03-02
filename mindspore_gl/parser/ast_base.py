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
"""Based AST Classes."""
import ast
from typing import List
from types import MethodType
from .backend import MindSporeBackend
from .gnn_exception import SyntaxNotSupported

NOT_SUPPORTED = {"NamedExp", "Set", "SetComp", "Await",
                 "Yield", "YieldFrom", "FormattedValue", "JoinedStr",
                 "With", "Import", "ImportFrom", "Global", "Nonlocal",
                 "Pass", "AnnAssign"}

SUPPORTED = {"For", "Assign", "FunctionDef", "Expr", "BinOp", "ListComp",
             "Call", "Attribute", "Name", "If", "Tuple"}

AGG_OP = {"sum", "max", "min", "avg"}

BACKEND = MindSporeBackend()


class SyntaxFilter:
    """
    Syntax Filter.
    """

    def __init__(self):
        self.init_unsupported_ast()
        self.agg_ops_ = AGG_OP
        self.supported_constructs_ = SUPPORTED

    def init_unsupported_ast(self):
        """
        Init unsupported ast.

        All ast in NOT_SUPPORTED will be prohibited.
        """

        def unsupported(node: ast.AST):
            """
            Prohibit the node.

            Args:
                node (ast.AST): the node to be prohibited.
            """
            raise SyntaxNotSupported(f"Line {node.lineno}: {node}")

        for ast_name in NOT_SUPPORTED:
            func_name = "visit_" + ast_name
            setattr(self, func_name, unsupported)


class BaseAstTransformer(ast.NodeTransformer, SyntaxFilter):
    """Base Ast Transformer."""

    def __init__(self):
        super().__init__()
        self.backend_ = BACKEND

    def ordered_visit(self, node, field_list: List[str]):
        """
        Ordered visit the node the in field list.

        Args:
            node (ast.AST): the node visiting.
            field_list (List[str]): List of fields in the node
            which need visiting.

        Returns:
            ast.AST, the node after modification.
        """
        for field in field_list:
            old_value = getattr(node, field)
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node

    def __repr__(self):
        return str(self.__class__)


class BaseAstVisitor(ast.NodeVisitor, SyntaxFilter):
    """Base Ast Visitor."""

    def __init__(self):
        super().__init__()
        self.ast_transformation_ = {}

    def ordered_visit(self, node, field_list: List[str]):
        """
        Ordered visit the node the in field list.

        Args:
            node (ast.AST): the node visiting.
            field_list (List[str]): List of fields in the node
            which need visiting.
        """
        for field in field_list:
            value = getattr(node, field)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)

    def add_transformation(self,
                           node: ast.AST,
                           transformation: BaseAstTransformer):
        """
        Add a transformation for the node.

        Args:
            node (ast.AST): the node visiting.
            transformation (BaseAstTransformer):
            the transformation to be added.
        """
        if node in self.ast_transformation_:
            self.ast_transformation_[node].append(transformation)
        else:
            self.ast_transformation_[node] = [transformation]


class FindChild(BaseAstVisitor):
    """Find a particular child in the node."""

    def __init__(self):
        super().__init__()
        self.found_ = False

        def match(self, node: ast.AST):
            """
            Determine whether the node was found or not.

            Args:
                node (ast.AST): the node to be found.
            """
            self.generic_visit(node)
            if not self.found_:
                self.found_ = (node == self.target_)

        for func_name in self.supported_constructs_:
            setattr(self, "visit_" + func_name, MethodType(match, self))

    def find_child(self, node: ast.AST, target: ast.AST):
        """
        Find a particular child in a node.

        Args:
            node (ast.AST): the node visiting.
            target (ast.AST): the node to be found.

        Returns:
            bool, whether the target was found in the node.
        """
        self.found_ = False
        self.target_ = target
        self.generic_visit(node)
        return self.found_
