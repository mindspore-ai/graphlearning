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
"""Code comparator."""
import ast
from ast_decompiler import decompile


def trace_stmt_insertion(nb_stmt: ast.AST, new_stmt: ast.AST):
    """
    Trace the statement insertion.

    If a statement is inserted and it does not have a correct lineno,
    use this func to track the insertion.

    Args:
        nb_stmt (ast.AST): the neighbour statement.
        new_stmt (ast.AST): the new statement.
    """
    new_stmt.lineno = nb_stmt.lineno
    if hasattr(nb_stmt, "origin_lineno"):
        new_stmt.origin_lineno = nb_stmt.origin_lineno


def trace_stmt_replacement(old_stmt: ast.AST, new_stmt: ast.AST):
    """
    Trace the statement replacement.

    Original lineno of old ast should be copied
    when it's replaced by a new ast.

    Args:
        old_stmt (ast.AST): the old statement.
        new_stmt (ast.AST): the new statement.
    """
    if hasattr(old_stmt, "origin_lineno"):
        new_stmt.origin_lineno = old_stmt.origin_lineno


class CodeComparator:
    """Code comparator class."""

    def __init__(self, screen_width):
        self.screen_width = screen_width
        self.vis_width = int(screen_width / 2 - 7)
        self.old_line_list = []
        self.new_line_list = []
        self.line2line_map = {}

    def record_origin_lineno(self, py_ast):
        """
        Record the origin lineno information. Use as load the old ast.

        Args:
            py_ast (ast.AST): the origin ast.
        """
        self.generate_line_map(
            py_ast, self.old_line_list, '', self.handle_record_lineno)

    def mapping_by_origin_lineno(self, py_ast):
        """
        Generate a line to line map according to origin lineno.
        Use as load the new ast.

        Args:
            py_ast (ast.AST): the new ast.
        """
        self.generate_line_map(
            py_ast, self.new_line_list, '', self.handle_mapping_by_lineno)

    def show_diff(self):
        """
        Show the difference according to the old ast and the new ast.
        """
        print('-' * self.screen_width)
        self.display_old_lineno, self.display_new_lineno = 1, 1
        old_idx, new_idx = 0, 0

        max_old_idx = len(self.old_line_list)
        max_new_idx = len(self.new_line_list)
        while old_idx < max_old_idx and new_idx < max_new_idx:
            (old_lineno, old_context) = self.old_line_list[old_idx]
            (new_lineno, new_context) = self.new_line_list[new_idx]
            if self.line2line_map[new_lineno] == old_lineno:
                self.print_line(old_context, new_context)
                old_idx += 1
                new_idx += 1
            elif self.line2line_map[new_lineno] < old_lineno:
                self.print_line('', new_context)
                new_idx += 1
            else:
                self.print_line(old_context, '')
                old_idx += 1
        while old_idx < max_old_idx:
            (old_lineno, old_context) = self.old_line_list[old_idx]
            self.print_line(old_context, '')
            old_idx += 1
        while new_idx < max_new_idx:
            (new_lineno, new_context) = self.new_line_list[new_idx]
            self.print_line('', new_context)
            new_idx += 1
        print('-' * self.screen_width)

    def print_line(self, old_context, new_context):
        """
        Print a single line.

        Args:
            old_context (str): the old ast context.
            new_context (str): the new ast context.
        """
        old_ctx_list = old_context.split('\n')
        new_ctx_list = new_context.split('\n')
        list_length = max(len(old_ctx_list), len(new_ctx_list))
        old_ctx_list.extend('' for _ in range(list_length - len(old_ctx_list)))
        new_ctx_list.extend('' for _ in range(list_length - len(new_ctx_list)))
        intermediate = ''
        if old_ctx_list[0] != '':
            intermediate += str(self.display_old_lineno).center(4)
            self.display_old_lineno += 1
        else:
            intermediate += '    '
        intermediate += ' || '
        if new_ctx_list[0] != '':
            intermediate += str(self.display_new_lineno).center(4)
            self.display_new_lineno += 1
        else:
            intermediate += '    '
        for i in range(list_length):
            if len(old_ctx_list[i]) > self.vis_width:
                old_ctx_list[i] = old_ctx_list[i][:self.vis_width - 3] + '...'
            if len(new_ctx_list[i]) > self.vis_width:
                new_ctx_list[i] = new_ctx_list[i][:self.vis_width - 3] + '...'
            print('|' + old_ctx_list[i].ljust(self.vis_width) + intermediate +
                  new_ctx_list[i].ljust(self.vis_width) + '|')
            intermediate = '     ||     '

    def generate_line_map(self,
                          py_ast: ast.AST,
                          line_list: list,
                          prefix: str,
                          ast_handle_cb):
        """Generate a line map."""
        if self.is_invisible_line(py_ast):
            return
        if hasattr(py_ast, 'lineno'):
            line_list.append((py_ast.lineno, self.node2string(py_ast, prefix)))
            ast_handle_cb(py_ast)
        if hasattr(py_ast, 'body'):
            for item in py_ast.body:
                self.generate_line_map(
                    item, line_list, prefix + '    ', ast_handle_cb)
        if hasattr(py_ast, 'orelse'):
            if not py_ast.orelse:
                return
            if len(py_ast.orelse) == 1 and\
               isinstance(py_ast.orelse[0], ast.If):
                py_ast.orelse[0].iselse = True
                self.generate_line_map(
                    py_ast.orelse[0], line_list, prefix, ast_handle_cb)
            else:
                line_list.append((py_ast.orelse[0].lineno, prefix + 'else:'))
                for item in py_ast.orelse:
                    self.generate_line_map(
                        item, line_list, prefix + '    ', ast_handle_cb)

    def handle_record_lineno(self, node):
        """
        Handler for the origin ast.

        Args:
            node (ast.AST): the node to be handled.
        """
        node.origin_lineno = node.lineno

    def handle_mapping_by_lineno(self, node):
        """
        Handler for the new ast.

        Args:
            node (ast.AST): the node to be handled.
        """
        if node.lineno in self.line2line_map and \
           self.line2line_map[node.lineno] != -1:
            return
        origin_lineno = -1
        if hasattr(node, 'origin_lineno'):
            origin_lineno = node.origin_lineno
        self.line2line_map[node.lineno] = origin_lineno

    def is_invisible_line(self, py_ast):
        """
        Determine whether this line should be shown in comparator.

        Some line should not be shown and here
        will record. e.g. Lines tag in \"\".
        Developers can add new judgement here if other cases found.

        Args:
            py_ast (ast.AST): Current python ast node.

        Returns:
            bool, whether it is a invisible line.
        """
        if isinstance(py_ast, ast.Expr):
            if isinstance(py_ast.value, ast.Str):
                return True
        return False

    def node2string(self, py_ast, prefix):
        """
        Node to string transformation.

        Args:
            py_ast (ast.AST): ast node.
            prefix (str): prefix empty spaces.

        Returns:
            str, string after transformation.
        """
        context = decompile(
            py_ast, line_length=self.vis_width - len(prefix)).strip()
        if hasattr(py_ast, 'iselse'):
            context = 'el' + context
        if ':\n' in context:
            context = context.split(':\n', maxsplit=1)[0] + ':'
        context = prefix + context.replace('\n', '\n' + prefix)
        return context
