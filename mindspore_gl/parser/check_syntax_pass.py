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
"""Check syntax pass."""
import ast
import types
from collections.abc import Iterable
from .backend import set_backend
from .scope import ScopeManager, Scoped
from .ast_base import BaseAstVisitor
from .api import Graph, BatchedGraph, HeterGraph, Edge, SrcVertex, DstVertex
from .gnn_exception import SyntaxNotSupported
from .vectorization import VectorizationType
from .supported_op import supported_ops, batchedgraph_ops_extend, hetergraph_ops_extend
from .ast_transformer import ArgReplacer, ArgListReplacer,\
    InitBackend, InlineAttributeSetter, InlineSupportedOpCall, \
    DefaultArgsSetter
from .constants import BATCHED_GRAPH_FIELD_NAMES, GRAPH_FIELD_NAMES, N_GRAPHS


class Symbol:
    """Based Symbol Class."""

    def __init__(self):
        self.is_built_in_ = False
        self.data_attr_ = {}
        self.data_attr_type_ = None

    def has_attr(self, attr_name):
        """Return True if it has the attribute."""
        return attr_name in self.data_attr_ or hasattr(self, attr_name)

    def has_data_attr(self, attr_name):
        """Return True if it has the data attribute."""
        return attr_name in self.data_attr_

    def has_func_attr(self, attr_name):
        """Return True if it has the function attribute."""
        return hasattr(self, attr_name) and callable(getattr(self, attr_name))

    def has_field_attr(self, attr_name):
        """Return True if it has the field attribute."""
        return hasattr(self, attr_name) and \
            not callable(getattr(self, attr_name))

    def get_attr_value(self, attr_name):
        """Get the attribute value."""
        if self.has_attr(attr_name):
            if hasattr(self, attr_name):
                attr_value = getattr(self, attr_name)
            else:
                attr_value = self.data_attr_[attr_name]
            return attr_value
        raise AttributeError(
            f"\"{self.__class__}\" object doesn't have attribute {attr_name}")

    def is_built_in(self):
        """Return True if it is a built in function."""
        return self.is_built_in_

    def set_data_attr(self, attr_name):
        """Set the data attribute."""
        if attr_name not in self.data_attr_:
            self.data_attr_[attr_name] = self.data_attr_type_
        else:
            assert self.data_attr_[attr_name] == self.data_attr_type_, \
                "Inconsistent vectorization type "\
                "for \"{}\", previous \"{}\", new \"{}\""

    def get_all_data_attr(self):
        """Get all data attributes."""
        return self.data_attr_


class SymOp(Symbol):
    """Op class."""

    def __init__(self):
        super().__init__()
        self.is_built_in_ = True

    def __call__(self):
        pass


class SymSrcVertex(Symbol):
    """Source Vertex Class."""

    def __init__(self):
        super().__init__()
        self.is_built_in_ = True
        self.class_ = SrcVertex
        self.data_attr_type_ = VectorizationType.SRC


class SymDstVertex(Symbol):
    """Destination Vertex Class."""

    def __init__(self, nb):
        super().__init__()
        self.innbs = nb
        self.is_built_in_ = True
        self.class_ = DstVertex
        self.data_attr_type_ = VectorizationType.DST


class SymEdge(Symbol):
    """Edge Class."""

    def __init__(self, src, dst):
        super().__init__()
        self.src = src
        self.dst = dst
        dst.inedges = (src, self)
        self.is_built_in_ = True
        self.class_ = Edge
        self.data_attr_type_ = VectorizationType.EDGE


class SymBaseGraph(Symbol):
    """
    Symbol Base Graph class.

    Abstract class for Graph class which needs translate.
    """

    def __init__(self):
        super().__init__()
        self.is_graph = True
        self.is_built_in_ = False
        self.data_attr_type_ = VectorizationType.GRAPH
        self.fields_ = GRAPH_FIELD_NAMES

    def init_attr(self):
        """Init attribute."""
        for built_in_key in self.fields_:
            self.set_data_attr(built_in_key)


class SymGraph(SymBaseGraph):
    """Graph Class."""

    def __init__(self):
        super().__init__()
        self.src_vertex = SymSrcVertex()
        self.dst_vertex = SymDstVertex(self.src_vertex)
        self.edge = SymEdge(self.src_vertex, self.dst_vertex)
        self.is_built_in_ = True
        self.class_ = Graph
        self.init_attr()

    def _set_attr_call(self, call_node, target, func_name):
        """function for setting data attribute."""
        args = call_node.args
        if len(args) != 1 or not isinstance(args[0], ast.Dict):
            raise TypeError(
                f"Line {call_node.lineno}: built-in function \"{self.__class__}.{func_name}\" only"
                f" accepts a single dictionary type argument, but got {type(args[0])}.")
        for const_node in args[0].keys:
            if isinstance(const_node, ast.Str):
                target.set_data_attr(const_node.s)
                continue
            if isinstance(const_node, ast.Constant):
                #  Python 3.8 or above
                target.set_data_attr(const_node.value)
                continue
            raise SyntaxError(
                f"Line {call_node.lineno}: built-in function \"{self.__class__}.{func_name}\" only"
                f" accepts a single dictionary type argument with key of string type, but got {type(const_node)}.")

    def set_vertex_attr(self, call_node):
        """Graph function, set the vertex attributes."""
        self._set_attr_call(call_node, self.src_vertex, "set_vertex_attr")
        self._set_attr_call(call_node, self.dst_vertex, "set_vertex_attr")
        return VectorizationType.DST

    def set_src_attr(self, call_node):
        """Graph function, set the source vertex attributes."""
        self._set_attr_call(call_node, self.src_vertex, "set_src_attr")
        return VectorizationType.SRC

    def set_dst_attr(self, call_node):
        """Graph function, set the destination vertex attributes."""
        self._set_attr_call(call_node, self.dst_vertex, "set_dst_attr")
        return VectorizationType.DST

    def set_edge_attr(self, call_node):
        """Graph function, set the edge attributes."""
        self._set_attr_call(call_node, self.edge, "set_edge_attr")
        return VectorizationType.EDGE

    def set_graph_attr(self, call_node):
        """Graph function, set the graph attributes."""
        self._set_attr_call(call_node, self, "set_graph_attr")
        return VectorizationType.GRAPH


class SymBatchedGraph(SymGraph):
    """Batched Graph Class."""

    def __init__(self):
        super().__init__()
        self.class_ = BatchedGraph
        self.fields_ = BATCHED_GRAPH_FIELD_NAMES
        self.init_attr()
        self.set_data_attr(N_GRAPHS)


class SymHeterGraph(SymBaseGraph):
    """Batched Graph Class."""

    def __init__(self):
        super().__init__()
        self.class_ = HeterGraph
        self.is_built_in_ = True
        self.init_attr()


TYPE_GRAPH = "Graph"
TYPE_BATCHEDGRAPH = "BatchedGraph"
TYPE_HETERGRAPH = "HeterGraph"
TYPE_DSTVERTEX = "DstVertex"
TYPE_SRCVERTEX = "SrcVertex"
TYPE_EDGE = "TypeEdge"
TYPE_OP = "Op"

BUILT_IN_CLASSES = {
    TYPE_GRAPH: SymGraph,
    TYPE_BATCHEDGRAPH: SymBatchedGraph,
    TYPE_HETERGRAPH: SymHeterGraph,
    TYPE_DSTVERTEX: SymDstVertex,
    TYPE_SRCVERTEX: SymSrcVertex,
    TYPE_EDGE: SymEdge,
    TYPE_OP: SymOp,
}


class SymTable(Scoped):
    """Symbol Table class."""

    def __init__(self, globals_dict):
        super().__init__()
        assert isinstance(globals_dict, dict)
        self.sym_table_impl_ = [{key: Symbol() for key in globals_dict}]

    def add_symbol(self, sym_id, class_name=None):
        """Add a symbol to the table."""
        if class_name and class_name in BUILT_IN_CLASSES:
            self.sym_table_impl_[-1][sym_id] = BUILT_IN_CLASSES[class_name]()
        else:
            self.sym_table_impl_[-1][sym_id] = Symbol()

    def set_symbol(self, sym_id, class_instance):
        """Set a symbol in the table."""
        self.sym_table_impl_[-1][sym_id] = class_instance

    def lookup(self, sym_id):
        """Look up a symbol in the table by it's id."""
        for sym_table in self.sym_table_impl_[::-1]:
            if sym_id in sym_table:
                return sym_table[sym_id]
        return None

    def items(self):
        """Get the symbol items."""
        for table in self.sym_table_impl_:
            for sym_id, sym in table.items():
                yield sym_id, sym

    def new_scope(self):
        """Create a new scope."""
        self.sym_table_impl_.append({})
        return self.sym_table_impl_[-1]

    def exit_scope(self):
        """Exit the current scope."""
        self.sym_table_impl_.pop()


class AttributeRecursionDepth(Scoped):
    """Attribute Recursion Depth class."""

    def __init__(self):
        super().__init__()
        self.recursion_depth = 0

    def new_scope(self):
        """Create a new scope."""
        self.recursion_depth += 1
        return self.recursion_depth

    def exit_scope(self):
        """Exit the current scope."""
        self.recursion_depth -= 1


class CheckSyntaxPass(BaseAstVisitor):
    """Check Syntax Pass Class."""

    def __init__(self, globals_dict):
        super().__init__()
        set_backend(self.find_backend(globals_dict))
        self.sym_table_ = SymTable(globals_dict)
        self.attr_recursion_ = AttributeRecursionDepth()
        self.func_def_count_ = 0
        self.expr_graph_type_ = {}
        self.graph_type_ = None
        self.add_builtin_ops_to_symtable([op_name for op_name in self.agg_ops_] + ["compile"])
        setattr(self.__class__, "visit_FunctionDef", self.visit_functiondef)
        setattr(self.__class__, "visit_Expr", self.visit_expr)
        setattr(self.__class__, "visit_For", self.visit_for)
        setattr(self.__class__, "visit_ListComp", self.visit_listcomp)
        setattr(self.__class__, "visit_Attribute", self.visit_attribute)
        setattr(self.__class__, "visit_Name", self.visit_name)
        setattr(self.__class__, "visit_Call", self.visit_call)

    def analyze(self, py_ast):
        """Analyze the ast tree."""
        self.visit(py_ast)
        return self.expr_graph_type_, self.ast_transformation_

    def visit_functiondef(self, node):
        """Visit ast.FunctionDef."""
        self.func_def_count_ += 1
        if self.func_def_count_ == 1:
            pos, graph_sym, self.graph_type_ = self.add_args_to_symtable(node.args)
            if isinstance(graph_sym, SymGraph):
                self.add_transformation(node, InitBackend(self.graph_type_))
            self.add_transformation(node, ArgReplacer(graph_sym.fields_, pos))
            if self.graph_type_ == TYPE_GRAPH:
                self.add_transformation(
                    node,
                    DefaultArgsSetter(
                        len(BATCHED_GRAPH_FIELD_NAMES)
                        - len(graph_sym.fields_)))
        self.generic_visit(node)

    def visit_expr(self, node):
        """Visit ast.Expr."""
        if self.is_built_in_call(node.value):
            node_type = self.handle_built_in_call(node.value)
            if isinstance(node_type, VectorizationType):
                self.expr_graph_type_[node] = VectorizationType.GRAPH
                self.add_transformation(node, InlineAttributeSetter(node_type))
        else:
            self.generic_visit(node)

    def visit_for(self, node):
        """Visit ast.For."""
        with ScopeManager(self.sym_table_) as _:
            # Visit iter and target first to
            # verify validness of symbols, attributes and etc
            self.ordered_visit(node, ["iter", "target"])
            target_node = node.target
            iter_node = node.iter
            if self.is_built_in_iter(iter_node):
                iter_sym = self.find_symbol(iter_node.value.id)
                self.set_symbol(
                    target_node.id, iter_sym.get_attr_value(iter_node.attr))
                self.expr_graph_type_[iter_node] = self.iter_graph_type(
                    iter_node)
            self.ordered_visit(node, ["body", "orelse"])

    def visit_listcomp(self, node):
        """Visit ast.ListComp."""
        # We force visiting generators before
        # visiting elt to make sure we write the symbol
        # into sym table before read
        with ScopeManager(self.sym_table_) as _:
            self.ordered_visit(node, ["generators"])
            for comp in node.generators:
                target_node = comp.target
                iter_node = comp.iter
                if self.is_built_in_iter(iter_node):
                    iter_sym = self.find_symbol(iter_node.value.id)
                    if isinstance(target_node, ast.Tuple):
                        symbols = iter_sym.get_attr_value(iter_node.attr)
                        assert isinstance(symbols, Iterable), \
                               f"Line {iter_node.lineno}:{iter_node.value.id}." \
                               f"{iter_node.attr} cannot be unpacked to {[elt.id for elt in target_node.elts]}"
                        if len(symbols) != len(target_node.elts):
                            raise SyntaxError(f"Expected tuple size:{len(symbols)}," \
                                              f" Actual tuple size:{len(target_node.elts)}.")
                        for elt, sym in zip(target_node.elts, symbols):
                            self.set_symbol(elt.id, sym)
                    elif isinstance(target_node, ast.Name):
                        self.set_symbol(
                            target_node.id, iter_sym.get_attr_value(
                                iter_node.attr))
                    self.expr_graph_type_[iter_node] = self.iter_graph_type(
                        iter_node)
            self.ordered_visit(node, ["elt"])

    def visit_attribute(self, node):
        """Visit ast.Attribute."""
        with ScopeManager(self.attr_recursion_) as recur_depth:
            self.generic_visit(node)
            if isinstance(node.value, ast.Name):
                sym = self.find_symbol(node.value.id)
                if not sym:
                    raise SyntaxError(
                        f"Cannot find symbol of {node.value.id} in Line {node.lineno}")
                if sym.is_built_in():
                    if recur_depth > 1:
                        raise SyntaxError(
                            f"Line {node.lineno}: \"{node.value.id}\" object \"{sym.__class__}\" doesn't support"
                            " accessing attribute recursively.")
                    if isinstance(node.ctx, ast.Load):
                        if not sym.has_attr(node.attr) and \
                           not hasattr(sym.class_, node.attr):
                            raise AttributeError(
                                f"Line {node.lineno}: No attribute \"{node.attr}\""
                                f" for variable \"{node.value.id}\" of class \"{sym.__class__}\"")
                        # Only built-in symbols' data attribute
                        # are initialized with tensor type
                        if sym.has_data_attr(node.attr):
                            node_type = sym.get_attr_value(node.attr)
                            self.expr_graph_type_[node] = node_type
                    else:
                        # ast.Store()
                        if sym.has_func_attr(node.attr) \
                           or sym.has_field_attr(node.attr):
                            raise AttributeError(
                                f"Line {node.lineno}: cannot override built-in"
                                f" attribute name \"{node.attr}\" for {sym.__class__} object")
                        sym.set_data_attr(node.attr)
                        self.expr_graph_type_[node] = sym.get_attr_value(
                            node.attr)

    def visit_name(self, node):
        """Visit ast.Name."""
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Load):
            sym = self.find_symbol(node.id)
            if not sym:
                print(NameError(
                    f"WARNING: Line {node.lineno}: \"{node.id}\" is undefined"))
        elif isinstance(node.ctx, ast.Store):
            self.add_symbol(node.id)
        return node

    def visit_call(self, node):
        """Visit ast.Call."""
        self.generic_visit(node)
        for i, arg in enumerate(node.args):
            if isinstance(arg, ast.Name):
                sym = self.find_symbol(arg.id)
                if isinstance(sym, SymBaseGraph):
                    self.add_transformation(
                        node, ArgListReplacer(sym.fields_, i))
        if self.is_built_in_call(node):
            self.handle_built_in_call(node)
        return node

    def is_built_in_iter(self, iter_node):
        """Helper function, return True if it is built in iter."""
        if isinstance(iter_node, ast.Attribute) \
           and isinstance(iter_node.value, ast.Name):
            iter_sym = self.find_symbol(iter_node.value.id)
            if iter_sym.is_built_in():
                if iter_sym.has_field_attr(iter_node.attr):
                    return True
                raise SyntaxError(
                    f"Line {iter_node.lineno}: \"{iter_sym.__class__}\" "
                    f"doesn't have attribute \"{iter_node.attr}\".")
        return False

    def find_backend(self, globals_dict):
        """Find the backend name."""
        backends = ["mindspore"]
        bk_name = None
        for k, v in globals_dict.items():
            if not isinstance(v, types.ModuleType):
                continue
            for bk in backends:
                if bk == v.__name__:
                    bk_name = k
            if bk_name:
                break
        if not bk_name:
            raise AttributeError(
                f"None of backend from {backends} is identified."
                f" Backend must be imported as a global variable.")
        return bk_name

    def iter_graph_type(self, iter_node):
        """Iter graph type."""
        iter_sym = self.find_symbol(iter_node.value.id)
        if isinstance(iter_sym, SymBaseGraph):
            return VectorizationType.VERTEX
        if isinstance(iter_sym, SymDstVertex):
            return VectorizationType.EDGE
        raise SyntaxError(
            f"Line {iter_node.lineno}: cannot figure out VectorizationType of\"{iter_node.value.id}\".")

    def add_args_to_symtable(self, arguments):
        """Add the arguments to the symbol table."""
        missing_graph = True
        for i, arg in enumerate(arguments.args):
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    arg_type = arg.annotation.id
                else:
                    arg_type = "Unknown"
                self.add_symbol(arg.arg, arg_type)
                self.add_symbol(arg_type)
                if TYPE_GRAPH in arg_type:
                    missing_graph = False
                    graph_type = arg_type
                    graph = self.find_symbol(arg.arg)
                    pos = i
            else:
                self.add_symbol(arg.arg)
        if missing_graph:
            raise SyntaxError(
                f"One of the arguments of vertex-centric "
                f"FunctionDef must be annotated with \"{TYPE_GRAPH}\".")
        return pos, graph, graph_type

    def is_built_in_call(self, call_node):
        """Return True if the call is built in call."""
        if (isinstance(call_node, ast.Call) and
                isinstance(call_node.func, ast.Attribute) and
                isinstance(call_node.func.value, ast.Name)):
            func = call_node.func
            func_name = func.attr
            symbol = self.find_symbol(func.value.id)
            if symbol.is_built_in():
                if hasattr(symbol.class_, func_name) \
                   and callable(getattr(symbol.class_, func_name)):
                    return True
                raise AttributeError(
                    f"Line {call_node.lineno}: {symbol.__class__} class doesn't have function {func_name}.")
        return False

    def handle_built_in_call(self, call_node):
        """Handle the built in call node."""
        func = call_node.func
        if not isinstance(func, ast.Attribute):
            raise SyntaxNotSupported(
                "Built-in function must be accessed"
                " as attribute of built-in class")
        sym = self.find_symbol(func.value.id)
        assert sym, "Cannot resolve symbol " + str(func.value.id)
        method_name = func.attr
        if self.graph_type_ == TYPE_BATCHEDGRAPH:
            supported_ops_ = {**supported_ops, **batchedgraph_ops_extend}
        elif self.graph_type_ == TYPE_HETERGRAPH:
            supported_ops_ = {**supported_ops, **hetergraph_ops_extend}
        else:
            supported_ops_ = supported_ops
        if method_name in supported_ops_:
            # supported Op functions
            self.expr_graph_type_[call_node] = VectorizationType.GRAPH
            self.add_transformation(
                call_node, InlineSupportedOpCall(
                    supported_ops_[method_name]))
            return None

        # set attribute functions
        return sym.get_attr_value(method_name)(call_node)

    def add_builtin_ops_to_symtable(self, op_name_list):
        """Add the built in ops to the symbol table."""
        for op_n in op_name_list:
            self.add_symbol(op_n, TYPE_OP)

    def add_symbol(self, sym_id, class_name=None):
        """Add a symbol to the table."""
        self.sym_table_.add_symbol(sym_id, class_name)

    def set_symbol(self, sym_id, class_instance):
        """Set a symbol in the symbol table."""
        self.sym_table_.set_symbol(sym_id, class_instance)

    def find_symbol(self, sym_id):
        """Find a symbol in the symbol table by it's id."""
        return self.sym_table_.lookup(sym_id)
