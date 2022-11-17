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
"""Alchemy Dataset"""
from typing import Optional, Union
import pathlib
from collections import defaultdict
import numpy as np
from mindspore_gl.graph import MindHomoGraph
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import pandas as pd
from tqdm import tqdm

class Alchemy:
    """
    Alchemy dataset, a source dataset for reading and parsing Alchemy dataset.

    Args:
        root(str): path to the root directory that contains alchemy_with_mask.npz.

    Raises:
        TypeError: if `root` is not a str.
        RuntimeError: if `root` does not contain data files.
        ValueError: if `datasize` is more than 99776.

    Examples:
        >>> from mindspore_gl.dataset import Alchemy
        >>> root = "path/to/alchemy"
        >>> dataset = Alchemy(root)

    About Alchemy dataset:
    The Tencent Quantum Lab has recently introduced a new molecular dataset, called Alchemy, to facilitate the
    development of new machine learning models useful for chemistry and materials science.

    The dataset lists 12 quantum mechanical properties of 130,000+ organic molecules comprising up to 12 heavy atoms
    (C, N, O, S, F and Cl), sampled from the GDBMedChem database. These properties have been calculated using the
    open-source computational chemistry program Python-based Simulation of Chemistry Framework (PySCF).

    Statistics:

    - Graphs: 99776
    - Nodes: 9.71
    - Edges: 10.02
    - Number of quantum mechanical properties: 12

    Dataset can be download here: <https://alchemy.tencent.com/data/dev_v20190730.zip> & <https://alchemy.tencent.com/data/valid_v20190730.zip>
    You can organize the dataset files into the following directory structure and read by `preprocess` API.

    .. code-block::
    .
    ├── dev
    │ ├── dev_target.csv
    │ └── sdf
    │     ├── atom_10
    │     ├── atom_11
    │     ├── atom_12
    │     └── atom_9
    └── valid
        ├── sdf
        │ ├── atom_11
        │ └── atom_12
        └── valid_target.csv

    """
    dataset_url = ""
    fdef_name = pathlib.Path(RDConfig.RDDataDir) / 'BaseFeatures.fdef'
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(str(fdef_name))

    def __init__(self, root: Optional[str] = None, datasize=10000):
        if not isinstance(root, str):
            raise TypeError(f"For '{self.cls_name}', the 'root' should be a str, "
                            f"but got {type(root)}.")
        if datasize > 99776:
            raise ValueError(f"The maximum capacity of dataset is 99776")
        self._root = pathlib.Path(root)
        self._path = self._root / f'alchemy_{datasize}_with_mask.npz'.format(datasize)

        self._edge_array = None
        self._graphs = None

        self._node_feat = None
        self._edge_feat = None
        self._graph_label = None
        self._graph_nodes = None
        self._graph_edges = None

        self._train_mask = None
        self._val_mask = None
        self._datasize = datasize

        if self._root.is_dir() and self._path.is_file():
            self.load()
        elif self._root.is_dir():
            self.preprocess()
            self.load()
        else:
            raise Exception('data file does not exist')

    def preprocess(self):
        """process data"""
        node_feat_array, edges_feat_array, graph_label_array = None, None, None
        adj_coo_row, adj_coo_col = [], []
        graph_edges_list, graph_nodes_list = [0], [0]
        train_mask, val_mask = [], []
        for mode in ['dev', 'valid']:
            if mode == 'valid':
                tqdm_length = min(3951, self._datasize)
            else:
                tqdm_length = self._datasize
            pbar = tqdm(total=tqdm_length)

            target_file = self._root / mode / "{}_target.csv".format(mode)
            self.target = pd.read_csv(target_file, index_col=0,
                                      usecols=['gdb_idx',] + ['property_%d' % x for x in range(12)])
            self.target = self.target[['property_%d' % x for x in range(12)]]

            sdf_dir = self._root / mode / "sdf"
            sdf_list = sdf_dir.glob("**/*.sdf")
            atom_list = []
            for file in sdf_list:
                name = str(file)
                name = name[name.find('sdf/atom_'):].replace('sdf/atom_', '')
                name = int(name[:name.find('/')].replace('/', ''))
                atom_list.append([file, name])
            atom_list = sorted(atom_list, key=lambda x: x[1], reverse=True)
            search_list = [x[0] for x in atom_list]
            count = 0
            for sdf_file in search_list:
                if count >= tqdm_length:
                    break
                if mode == 'valid':
                    val_mask.append(1)
                    train_mask.append(0)
                else:
                    train_mask.append(1)
                    val_mask.append(0)
                count += 1
                num_atoms, node_feat, edges, edge_feat, label = self.file_to_graph(sdf_file)
                if edge_feat is None:
                    continue
                adj_coo_row += edges[0]
                adj_coo_col += edges[1]
                num_edges = len(edges[0])
                if node_feat_array is None:
                    node_feat_array = node_feat
                else:
                    node_feat_array = np.concatenate((node_feat_array, node_feat), axis=0)
                if edges_feat_array is None:
                    edges_feat_array = edge_feat
                else:
                    edges_feat_array = np.concatenate((edges_feat_array, edge_feat), axis=0)
                if graph_label_array is None:
                    graph_label_array = label
                else:
                    graph_label_array = np.concatenate((graph_label_array, label), axis=0)
                graph_nodes_list.append(num_atoms + graph_nodes_list[-1])
                graph_edges_list.append(num_edges + graph_edges_list[-1])
                pbar.update()
            pbar.close()
            print("loaded!")
        edge_array_list = np.array([adj_coo_row, adj_coo_col])
        np.savez(self._path, edge_array=edge_array_list, train_mask=train_mask, val_mask=val_mask,
                 node_feat=node_feat_array, edge_feat=edges_feat_array, graph_label=graph_label_array,
                 graph_edges=graph_edges_list, graph_nodes=graph_nodes_list)

    def file_to_graph(self, sdf_file):
        """
        Read sdf file and convert to feature data
        """
        sdf = open(str(sdf_file)).read()
        mol = Chem.MolFromMolBlock(sdf, removeHs=False)

        num_atoms = mol.GetNumAtoms()
        atom_feats = self.alchemy_nodes(mol)

        edges = [x for x in range(num_atoms) for y in range(num_atoms - 1)],\
                [y for x in range(num_atoms) for y in range(num_atoms) if x != y]

        bond_feats = self.alchemy_edges(mol)

        label = self.target.loc[int(sdf_file.stem)].tolist()
        label = np.array(label).reshape(1, -1)
        return num_atoms, atom_feats, edges, bond_feats, label

    def alchemy_nodes(self, mol):
        """
        Featurization for all atoms in a molecule
        """
        atom_feats = []
        donor_dict = defaultdict(int)
        acceptor_dict = defaultdict(int)
        def_file = str(pathlib.Path(RDConfig.RDDataDir) / 'BaseFeatures.fdef')
        mol_feats = ChemicalFeatures.BuildFeatureFactory(def_file).GetFeaturesForMol(mol)
        for molecule in mol_feats:
            if molecule.GetFamily() == 'Acceptor':
                atoms_list = molecule.GetAtomIds()
                for u in atoms_list:
                    acceptor_dict[u] = 1
            elif molecule.GetFamily() == 'Donor':
                atoms_list = molecule.GetAtomIds()
                for u in atoms_list:
                    donor_dict[u] = 1

        num_atoms = mol.GetNumAtoms()
        for u in range(num_atoms):
            atom = mol.GetAtomWithIdx(u)
            symbol = atom.GetSymbol()
            atomic_num = atom.GetAtomicNum()
            aromatic = atom.GetIsAromatic()
            hybridization = atom.GetHybridization()
            total_num_hs = atom.GetTotalNumHs()
            h_u = []
            h_u += [int(symbol == x) for x in ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']]
            h_u.append(atomic_num)
            h_u.append(acceptor_dict[u])
            h_u.append(donor_dict[u])
            h_u.append(int(aromatic))
            h_u += [int(hybridization == x) for x in (Chem.rdchem.HybridizationType.SP,
                                                      Chem.rdchem.HybridizationType.SP2,
                                                      Chem.rdchem.HybridizationType.SP3)
                    ]
            h_u.append(total_num_hs)
            atom_feats.append(h_u)
        return atom_feats

    def alchemy_edges(self, mol):
        """Featurization for all bonds in a molecule. The bond indices
        """
        edges_feats = []
        num_atoms = mol.GetNumAtoms()
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i == j:
                    continue
                e_uv = mol.GetBondBetweenAtoms(i, j)
                if e_uv is not None:
                    edges_type = e_uv.GetBondType()
                else:
                    edges_type = None
                edges_feats.append([
                    float(edges_type == x)
                    for x in (Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC,
                              None)
                ])
        return edges_feats


    def load(self):
        """Load the saved npz dataset from files."""
        self._npz_file = np.load(self._path, allow_pickle=True)
        self._edge_array = self._npz_file['edge_array'].astype(np.int64)
        self._graph_edges = self._npz_file['graph_edges'].astype(np.int64)
        self._graphs = np.array(list(range(len(self._graph_edges))))

    @property
    def num_features(self):
        """
        Feature size of each node

        Returns:
            int, the number of feature size

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> num_features = dataset.num_features
        """
        return self.node_feat.shape[-1]

    @property
    def num_edge_features(self):
        """
       Number of label classes

       Returns:
           int, the number of classes

       Examples:
           >>> #dataset is an instance object of Dataset
           >>> num_classes = dataset.num_classes
       """
        return self.edge_feat.shape[-1]

    @property
    def n_tasks(self):
        """
        Graph label size

        Returns:
            int, size of graph label

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> node_feat = dataset.graph_label
       """
        return self.graph_label.shape[-1]

    @property
    def train_mask(self):
        """
        Mask of training nodes

        Returns:
            numpy.ndarray, array of mask

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> train_mask = dataset.train_mask
        """
        if self._train_mask is None:
            self._train_mask = self._npz_file['train_mask']
        return self._train_mask

    @property
    def val_mask(self):
        """
        Mask of validation nodes

        Returns:
            numpy.ndarray, array of mask

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> val_mask = dataset.val_mask
        """
        if self._val_mask is None:
            self._val_mask = self._npz_file['val_mask']
        return self._val_mask

    @property
    def train_graphs(self):
        """
        Train graph id

        Returns:
            numpy.ndarray, array of train graph id

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> train_graphs = dataset.train_graphs
        """
        return (np.nonzero(self.train_mask)[0]).astype(np.int32)

    @property
    def val_graphs(self):
        """
        Valid graph id

        Returns:
            numpy.ndarray, array of valid graph id

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> val_graphs = dataset.val_graphs
        """
        return (np.nonzero(self.val_mask)[0]).astype(np.int32)

    @property
    def graph_nodes(self):
        """
        Accumulative graph nodes count

        Returns:
            numpy.ndarray, array of accumulative nodes

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> val_mask = dataset.graph_nodes
        """
        if self._graph_nodes is None:
            self._graph_nodes = self._npz_file['graph_nodes']
        return self._graph_nodes

    @property
    def graph_edges(self):
        """
        Accumulative graph edges count

        Returns:
            numpy.ndarray, array of accumulative edges

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> val_mask = dataset.graph_edges
        """
        if self._graph_edges is None:
            self._graph_edges = self._npz_file['graph_edges']
        return self._graph_edges

    @property
    def graph_count(self):
        """
        Total graph numbers

        Returns:
            int, numbers of graph

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> node_feat = dataset.node_feat
        """
        return len(self._graphs)

    @property
    def node_feat(self):
        """
        Node features

        Returns:
            numpy.ndarray, array of node feature

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> node_feat = dataset.node_feat
       """
        if self._node_feat is None:
            self._node_feat = self._npz_file["node_feat"]

        return self._node_feat

    @property
    def edge_feat(self):
        """
        Edge features

        Returns:
            numpy.ndarray, array of edge feature

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> node_feat = dataset.edge_feat
       """
        if self._edge_feat is None:
            self._edge_feat = self._npz_file["edge_feat"]

        return self._edge_feat

    def graph_feat(self, graph_idx):
        return self.node_feat[self.graph_nodes[graph_idx]: self.graph_nodes[graph_idx + 1]]

    def graph_edge_feat(self, graph_idx):
        return self.edge_feat[self.graph_edges[graph_idx]: self.graph_edges[graph_idx + 1]]

    @property
    def graph_label(self):
        """
        Graph label

        Returns:
            numpy.ndarray, array of graph label

        Examples:
            >>> #dataset is an instance object of Dataset
            >>> node_feat = dataset.graph_label
       """
        if self._graph_label is None:
            self._graph_label = self._npz_file["graph_label"]
        return self._graph_label

    def __getitem__(self, idx) -> Union[MindHomoGraph, np.ndarray]:
        assert idx < self.graph_count, "Index out of range"
        res = MindHomoGraph()
        # reindex to 0
        coo_array = self._edge_array[:, self.graph_edges[idx]: self.graph_edges[idx + 1]]
        res.set_topo_coo(coo_array)
        res.node_count = self.graph_nodes[idx + 1] - self.graph_nodes[idx]
        res.edge_count = self.graph_edges[idx + 1] - self.graph_edges[idx]
        return res
