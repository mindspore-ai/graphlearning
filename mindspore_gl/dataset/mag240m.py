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
"""MAG240M Dataset"""
#pylint: disable=W0702
from typing import Optional, Union, Dict
import os.path as osp
import numpy as np
from mindspore_gl.graph import MindRelationGraph, MindHeteroGraph, CsrAdj
from .utils import get_indptr_from_coo_src


class MAG240MDataset:
    """
    MAG240M Dataset, a source dataset for reading and parsing MAG240M dataset.

    Args:
        root(str): path to the root directory that contains mag240m data.

    Raises:
        TypeError: if `root` is not a str.
        RuntimeError: if `root` does not contain data files.

    Examples:
        >>> from mindspore_gl.dataset import MAG240MDataset
        >>> root = "path/to/mag240m"
        >>> dataset = MAG240MDataset(root)

    """

    __rels__ = {
        ('author', 'paper'): 'writes',
        ('author', 'institution'): 'affiliated_with',
        ('paper', 'paper'): 'cites',
    }

    def __init__(self, root: str = 'dataset'):
        if isinstance(root, str):
            root = osp.expanduser(osp.normpath(root))
        else:
            raise TypeError(f"For '{self.cls_name}', the 'root' should be a str, "
                            f"but got {type(root)}.")
        self.root = root
        self.dir = osp.join(root, 'mag240m')

        # load meta and split
        self.__meta__ = np.load(osp.join(self.dir, "meta.npz"))
        self.__split__ = np.load(osp.join(self.dir, "split.npz"))
        self.__full_topo_csr__ = None
        self.__full_feats__ = None

    @property
    def num_papers(self) -> int:
        return int(self.__meta__['paper'])

    @property
    def num_authors(self) -> int:
        return int(self.__meta__['author'])

    @property
    def num_institutions(self) -> int:
        return int(self.__meta__['institution'])

    @property
    def num_paper_features(self) -> int:
        return 768

    @property
    def num_classes(self) -> int:
        return int(self.__meta__['num_classes'])

    def get_idx_split(
            self, split: Optional[str] = None
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        return self.__split__ if split is None else self.__split__[split]

    @property
    def paper_feat(self) -> np.ndarray:
        path = osp.join(self.dir, 'processed', 'paper', 'paper_feat.npy')
        return np.load(path, mmap_mode='r')

    @property
    def author_feat(self) -> np.ndarray:
        path = osp.join(self.dir, 'processed', 'paper', 'author_feat.npy')
        return np.load(path, mmap_mode='r')

    @property
    def institution_feat(self) -> np.ndarray:
        path = osp.join(self.dir, 'processed', 'paper', 'institution_feat.npy')
        return np.load(path, mmap_mode='r')

    @property
    def all_paper_feat(self) -> np.ndarray:
        path = osp.join(self.dir, 'processed', 'paper', 'paper_feat.npy')
        return np.load(path)

    @property
    def all_author_feat(self) -> np.ndarray:
        path = osp.join(self.dir, 'processed', 'paper', 'author_feat.npy')
        return np.load(path)

    @property
    def all_institution_feat(self) -> np.ndarray:
        path = osp.join(self.dir, 'processed', 'paper', 'institution_feat.npy')
        return np.load(path)

    @property
    def paper_label(self) -> np.ndarray:
        path = osp.join(self.dir, 'processed', 'paper', 'paper_label.npy')
        return np.load(path, mmap_mode='r')

    @property
    def all_paper_label(self) -> np.ndarray:
        path = osp.join(self.dir, 'processed', 'paper', 'paper_label.npy')
        return np.load(path)

    @property
    def paper_year(self) -> np.ndarray:
        path = osp.join(self.dir, 'processed', 'paper', 'paper_year.npy')
        return np.load(path, mmap_mode='r')

    @property
    def author_year(self) -> np.ndarray:
        path = osp.join(self.dir, 'processed', 'paper', 'author_year.npy')
        return np.load(path, mmap_mode='r')

    @property
    def institution_year(self) -> np.ndarray:
        path = osp.join(self.dir, 'processed', 'paper', 'institution_year.npy')
        return np.load(path, mmap_mode='r')

    @property
    def all_paper_year(self) -> np.ndarray:
        path = osp.join(self.dir, 'processed', 'paper', 'paper_year.npy')
        return np.load(path)

    @property
    def all_author_year(self) -> np.ndarray:
        path = osp.join(self.dir, 'processed', 'paper', 'author_year.npy')
        return np.load(path)

    @property
    def all_institution_year(self) -> np.ndarray:
        path = osp.join(self.dir, 'processed', 'paper', 'institution_year.npy')
        return np.load(path)

    def edge_index(self, id1: str, id2: str,
                   id3: Optional[str] = None) -> np.ndarray:
        src = id1
        rel, dst = (id3, id2) if id3 is None else (id2, id3)
        rel = self.__rels__[(src, dst)] if rel is None else rel
        name = f'{src}___{rel}___{dst}'
        path = osp.join(self.dir, 'processed', name, 'edge_index.npy')
        return np.load(path)

    @property
    def full_topo_csr(self):
        return 1

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def __getitem__(self, graph_idx) -> MindHeteroGraph:
        assert graph_idx == 0, "MAG240M only has one graph"

        result_graph = MindHeteroGraph()

        for k, e_type in self.__rels__:
            src_type, dst_type = k
            relation_edge_index = self.edge_index(src_type, e_type, dst_type)
            relation_edge_index = relation_edge_index.astype(np.int32)
            relation_graph = MindRelationGraph(src_node_type=src_type, dst_node_type=dst_type, edge_type=e_type)
            indptr = np.zeros(relation_edge_index[1][-1] + 1, dtype=np.int32)
            get_indptr_from_coo_src(relation_edge_index[0], indptr)
            adj = CsrAdj(
                indptr=indptr,
                indices=relation_edge_index[1]
            )
            relation_graph.set_topo(adj_csr=adj)
            result_graph.add_graph(relation_graph)

        return result_graph
