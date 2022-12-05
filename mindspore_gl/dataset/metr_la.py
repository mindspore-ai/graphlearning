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
"""METR LA"""
import os
import numpy as np

class MetrLa:
    """
    METR-LA is a large-scale dataset collected from 1500 traffic loop detectors in
    Los Angeles country road network. This dataset includes speed, volume and occupancy
    data, covering approximately 3,420 miles.

    Args:
        root(str): path to the root directory that contains METR-LA/adj_mat.npy and
            METR-LA/node_values.npy.

    Inputs:
        - **in_timestep** (int) - numbers of input time sequence.
        - **out_timestep** (int) - numbers of output time sequence.

    Raises:
        TypeError: if `root` is not a str.
        RuntimeError: if `root` does not contain data files.
        TypeError: If `in_timestep` or `out_timestep` is not a positive int.

    Examples:
        >>> from mindspore_gl.dataset.ppi import MetrLa
        >>> root = "path/to/metrla"
        >>> dataset = MetrLa(root)
        >>> features, labels = dataset.get_data(in_timestep, out_timestep)

    """
    def __init__(self, root):
        if not isinstance(root, str):
            raise TypeError(f"For '{self.cls_name}', the 'root' should be a str, "
                            f"but got {type(root)}.")
        self._root = root
        self._adj = os.path.join(root, 'adj_mat.npy')
        self._node = os.path.join(root, 'node_values.npy')
        self.load()

        if os.path.exists(self._adj) and os.path.isfile(self._adj) and \
            os.path.exists(self._node) and os.path.isfile(self._node):
            self.load()
        else:
            raise Exception('data file does not exist')

    def load(self):
        """load data"""
        self.adj = np.load(self._adj)
        index = np.nonzero(self.adj)
        self.edge_attr = self.adj[index]
        self.edge_index = np.stack(index, axis=0)
        self.x = np.load(self._node).transpose((1, 2, 0))

        means = np.mean(self.x, axis=(0, 2))
        self.x = self.x - means.reshape(1, -1, 1)
        stds = np.std(self.x, axis=(0, 2))
        self.x = self.x / stds.reshape(1, -1, 1)

    def get_data(self, in_timestep, out_timestep):
        """
        get sequence time feature and label

        Args:
            in_timestep(int): numbers of input time sequence.
            out_timestep(int): numbers of output time sequence.

        """
        if not (isinstance(in_timestep, int) and in_timestep > 0):
            raise Exception('the in_timestep must be a positive integer value')
        if not (isinstance(out_timestep, int) and out_timestep > 0):
            raise Exception('the out_timestep must be a positive integer value')

        indices = [(i, i + (in_timestep + out_timestep))
                   for i in range(self.x.shape[2] - (in_timestep + out_timestep) + 1)
                   ]
        features, labels = [], []
        for i, j in indices:
            features.append((self.x[:, :, i: i + in_timestep]))
            labels.append((self.x[:, 0, i + in_timestep: j]))
        self.features = np.array(features)
        self.labels = np.array(labels)

        return self.features, self.labels

    @property
    def node_num(self):
        """
        Number of nodes

        Returns:
            int, number of node

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> node_count = dataset.node_num
        """
        return self.features.shape[1]
