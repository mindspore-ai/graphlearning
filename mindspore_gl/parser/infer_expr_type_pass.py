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
"""Infer Expr Type Pass."""
import ast
from typing import Any, List

from .vectorization import VectorizationType
from .ast_base import BaseAstVisitor
from .gnn_exception import TypeInferenceError
from .ast_transformer import ReplaceName, CastGraphType, VectorizeFor,\
                             VectorizeListComp, FlattenAttribute


def infer_call_vtype(vec_types: List[VectorizationType]) -> VectorizationType:
    """Infer call vertex type."""
    assert vec_types, "Cannot infer vectorization"
    graph_rmvd = [v for v in vec_types
                  if v != VectorizationType.GRAPH
                  and v is not None]
    if not graph_rmvd:
        return VectorizationType.GRAPH
    if any(v == VectorizationType.EDGE for v in graph_rmvd):
        return VectorizationType.EDGE
    if all(v == graph_rmvd[0] for v in graph_rmvd):
        return graph_rmvd[0]
    return VectorizationType.EDGE


class InferExprTypePass(BaseAstVisitor):
    """Infer Expr Type Pass Class."""

    def __init__(self, analysis_res: tuple, src: str):
        super().__init__()
        self.type_dict_, self.ast_transformation_ = analysis_res
        self.src_ = src
        setattr(self.__class__, "visit_For", self.visit_for)
        setattr(self.__class__, "visit_Assign", self.visit_assign)
        setattr(self.__class__, "visit_ListComp", self.visit_listcomp)
        setattr(self.__class__, "visit_BinOp", self.visit_binop)
        setattr(self.__class__, "visit_Attribute", self.visit_attribute)
        setattr(self.__class__, "visit_Name", self.visit_name)
        setattr(self.__class__, "visit_Call", self.visit_call)

    def analyze(self, py_ast: ast.AST):
        """Analyze the ast tree."""
        self.visit(py_ast)
        return self.ast_transformation_

    def visit_for(self, node: ast.For):
        """Visit ast.For."""
        iter_node = node.iter
        if iter_node in self.type_dict_:
            if self.get_ret_vec_type(iter_node) == VectorizationType.VERTEX:
                self.type_dict_[node.target] = VectorizationType.DST
                self.generic_visit(node)
                del self.type_dict_[node.target]
                self.add_transformation(node, VectorizeFor())
        else:
            self.generic_visit(node)

    def visit_assign(self, node: ast.Assign):
        """Visit ast.Assign."""
        self.generic_visit(node)
        vec_type = self.get_ret_vec_type(node.value)
        for tar in node.targets:
            self.set_ret_vec_type(tar, vec_type)

    def visit_listcomp(self, node: ast.ListComp):
        """Visit ast.ListComp."""
        # TODO: handle multiple generators
        # Visit generators first to resolve Name's vec type
        self.ordered_visit(node, ["generators"])
        comp = node.generators[0]
        iter_vec_type = self.get_ret_vec_type(comp.iter)
        if iter_vec_type:
            self.set_ret_vec_type(node, iter_vec_type)
            # Essentially an assign statement
            self.type_dict_[comp.target] = iter_vec_type
            self.ordered_visit(node, ["elt"])
            # Delete when goes out of scope
            del self.type_dict_[comp.target]
            if self.get_ret_vec_type(node.elt) != iter_vec_type:
                self.add_transformation(
                    node.elt,
                    CastGraphType(
                        self.get_ret_vec_type(node.elt),
                        iter_vec_type))
            self.add_transformation(node, VectorizeListComp())
            if isinstance(comp.iter, ast.Name):
                self.add_transformation(
                    node.elt, ReplaceName(comp.target.id, comp.iter.id))

    def visit_binop(self, node: ast.BinOp) -> Any:
        """Visit ast.BinOp."""
        self.generic_visit(node)
        l_vtype = self.get_ret_vec_type(node.left)
        r_vtype = self.get_ret_vec_type(node.right)
        binop_type = infer_call_vtype([l_vtype, r_vtype])
        self.set_ret_vec_type(node, binop_type)
        if l_vtype in (VectorizationType.SRC,
                       VectorizationType.DST,
                       VectorizationType.EDGE) and l_vtype != binop_type:
            self.add_transformation(
                node.left, CastGraphType(l_vtype, binop_type))
        if l_vtype in (VectorizationType.SRC,
                       VectorizationType.DST,
                       VectorizationType.EDGE) and r_vtype != binop_type:
            self.add_transformation(
                node.right, CastGraphType(r_vtype, binop_type))

    def visit_call(self, node: ast.Call):
        """Visit ast.Call."""
        self.generic_visit(node)
        if self.is_agg_call(node):
            arg_vec_type = self.get_ret_vec_type(node.args[0])
            if len(node.args) > 1 or \
               arg_vec_type not in {VectorizationType.EDGE,
                                    VectorizationType.SRC}:
                raise TypeInferenceError(
                    f"Line {node.lineno}: Built-in agg func \"{node.func.attr}\" "
                    f"only takes expr of EDGE or SRC type. "
                    f"Got {arg_vec_type}.")
            # Agg op always returns DST type
            self.set_ret_vec_type(node, VectorizationType.DST)
        else:
            vec_types = [self.get_ret_vec_type(arg) for arg in node.args]
            vec_types += [self.get_ret_vec_type(kw.value)
                          for kw in node.keywords]
            if not vec_types:
                pass
            else:
                call_type = infer_call_vtype(vec_types)
                self.set_ret_vec_type(node, call_type)
                for arg in node.args:
                    arg_vec_type = self.get_ret_vec_type(arg)
                    if arg_vec_type != call_type:
                        self.add_transformation(
                            arg, CastGraphType(arg_vec_type, call_type))

    def visit_name(self, node: ast.Name):
        """Visit ast.Name."""
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Load):
            node_type = self.find_type_by_name_id(node.id)
            if node_type:
                self.set_ret_vec_type(node, node_type)

    def visit_attribute(self, node: ast.Attribute):
        """Visit ast.Attribute."""
        attr_type = self.get_ret_vec_type(node)
        if attr_type:
            self.add_transformation(node, FlattenAttribute())

    def is_agg_call(self, node: ast.Call) -> bool:
        """Helper Method, return True if it is a agg call."""
        if self.get_ret_vec_type(node) == VectorizationType.GRAPH and \
           isinstance(node.func, ast.Attribute) and \
           node.func.attr in self.agg_ops_:
            return True
        return False

    def get_ret_vec_type(self, node: ast.AST) -> VectorizationType:
        """Helper Method, return the vertex type."""
        if node in self.type_dict_:
            return self.type_dict_[node]
        return None

    def set_ret_vec_type(self, node: ast.AST, node_type: VectorizationType):
        """Helper Method, set the vertex type."""
        self.type_dict_[node] = node_type

    def find_type_by_name_id(self, name_id: str) -> VectorizationType:
        """Helper Method, find the type by it's name id."""
        for k, v in self.type_dict_.items():
            if isinstance(k, ast.Name) and k.id == name_id:
                if not v:
                    continue
                return v
        return None
