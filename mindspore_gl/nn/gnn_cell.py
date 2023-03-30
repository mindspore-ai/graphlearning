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
"""GNN Cell"""
from mindspore.nn import Cell
from ..parser.vcg import translate, set_display_config
from ..parser.backend import Backend
from ..parser.check_syntax_pass import SymBaseGraph
from ..parser.check_syntax_pass import CheckSyntaxPass
from ..backward import GatherNet, CSRReduceSumNet

class GNNCell(Cell):
    """
    GNN Cell class.

    Construct function will be translated by default.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    translate_path = None
    csr = False
    backward = False

    def __init__(self):
        super().__init__()
        Backend.csr = self.csr
        Backend.backward = self.backward
        SymBaseGraph.csr = self.csr
        CheckSyntaxPass.csr = self.csr
        translate(self, "construct", self.translate_path)
        # pylint: disable=C0103
        if self.csr:
            self.CSR_BACKWARD_GATHER = GatherNet()
            self.CSR_BACKWARD_REDUCE_SUM = CSRReduceSumNet()

    @staticmethod
    def enable_display(screen_width=200):
        """
        Enable display code comparison.

        Args:
            screen_width (int, optional): Determines the screen width on which the code is displayed. Default: 200.

        Examples:
            >>> from mindspore_gl.nn import GNNCell
            >>> GNNCell.enable_display(screen_width=350)
        """
        set_display_config(screen_width, True)

    @staticmethod
    def disable_display():
        """
        Disable display code comparison.

        Examples:
            >>> from mindspore_gl.nn import GNNCell
            >>> GNNCell.disable_display()
        """
        set_display_config(0, False)

    @classmethod
    def specify_path(cls, path):
        """
        Enable specify the construct file path.

        Args:
            path (str): The path for save the construct file.

        Examples:
            >>> from mindspore_gl.nn import GNNCell
            >>> GNNCell.specify_path('path/to/save')
        """
        cls.translate_path = path

    @classmethod
    def sparse_compute(cls, csr=False, backward=False):
        """
        Whether to use sparse operator to accelerate calculation.

        Args:
            csr (bool, optional): Is it a csr data structure. Default: False.
            backward (bool, optional): Whether to use custom back propagation. Default: False.

        Raises:
            ValueError: If `csr` is False and `backward` is True.

        Examples:
            >>> from mindspore_gl.nn import GNNCell
            >>> GNNCell.sparse_compute(csr=True, backward=False)
        """
        if csr is False and backward is True:
            ValueError("Custom back propagation is supported only when the data structure is CSR")
        cls.csr = csr
        cls.backward = csr and backward
