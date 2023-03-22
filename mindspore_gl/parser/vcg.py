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
"""Translation."""
import ast
import inspect
from types import MethodType
from textwrap import dedent
from ast_decompiler import decompile
from .infer_expr_type_pass import InferExprTypePass
from .check_syntax_pass import CheckSyntaxPass
from .ast_rewriter import AstRewriter
from .code_comparator import CodeComparator
from .utils import src_to_function

SCREEN_WIDTH = 200
DISPLAY = True


def set_display_config(screen_width, display):
    """
    Set screen width and display configure used for translate function.

    Args:
        screen_width (int): Determines the screen width on which the code is displayed.
        display (bool): Show code comparison or Not.
    """
    global SCREEN_WIDTH, DISPLAY
    SCREEN_WIDTH = screen_width
    DISPLAY = display


def translate(obj, method_name: str, translate_path: None or str = None):
    """
    Translate the vertex central code into MindSpore understandable code.

    After translation, a new function will generate in `/.mindspore_gl` .
    The origin method will be replaced with this function.

    Args:
        obj: (Object): The object.
        method_name (str): The name of the method to be translated.
        translate_path (str): The path for save the construct file. Default: None.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.ops as ops
        >>> from mindspore_gl.nn import GNNCell
        >>> from mindspore_gl import BatchedGraph
        >>> from mindspore_gl.parser.vcg import translate
        ...
        >>> class Net(GNNCell):
        ...     def __init__(self):
        ...         super().__init__()
        ...         translate(self, "loss")
        ...
        ...     def construct(self, pred, label, g: BatchedGraph):
        ...         loss = self.loss(pred, label, g)
        ...         loss = loss * g.graph_mask
        ...         return loss
        ...
        ...     def loss(self, pred, label, g: BatchedGraph):
        ...         criterion = ms.nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')
        ...         loss = criterion(pred, label)
        ...         loss = ops.ReduceMean()(loss * g.graph_mask)
        ...         return loss
    """
    global SCREEN_WIDTH, DISPLAY
    fn = getattr(obj, method_name)
    src = inspect.getsource(fn)
    src = dedent(src)
    py_ast = ast.parse(src)
    syntax_checker = CheckSyntaxPass(fn.__globals__)
    ret = syntax_checker.analyze(py_ast)
    type_inferer = InferExprTypePass(ret, src)
    ret = type_inferer.analyze(py_ast)
    if DISPLAY:
        comparator = CodeComparator(SCREEN_WIDTH)
        comparator.record_origin_lineno(py_ast)
    rewriter = AstRewriter(ret)
    new_ast = rewriter.visit(py_ast)
    if DISPLAY:
        comparator.mapping_by_origin_lineno(new_ast)
        comparator.show_diff()
    new_src = decompile(new_ast)
    new_fn = src_to_function(new_src, method_name, fn.__globals__, translate_path)
    new_fn.__module__ = fn.__module__
    setattr(obj, method_name, MethodType(new_fn, obj))
