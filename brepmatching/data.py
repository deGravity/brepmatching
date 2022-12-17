import json
import os
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
import xxhash
from automate import Part, PartFeatures, PartOptions, part_to_graph
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from .utils import zip_hetdata
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader


def make_match_data(zf, orig_path, var_path, match_path, include_meshes=True):
    options = PartOptions()
    if not include_meshes:
        options.tesselate = False
    with zf.open(match_path,'r') as f:
        matches = json.load(f)
    with zf.open(orig_path, 'r') as f:
        orig_part = Part(f.read().decode('utf-8'), options)
    if not orig_part.is_valid:
        return None
    with zf.open(var_path, 'r') as f:
        var_part = Part(f.read().decode('utf-8'), options)
    if not var_part.is_valid:
        return None
    features = PartFeatures()
    if not include_meshes:
        features.mesh = False
    orig_brep = part_to_graph(orig_part, features)
    var_brep = part_to_graph(var_part, features)
    export_id_hash = lambda x: xxhash.xxh32(x).intdigest()
    hashed_matches = [(export_id_hash(match['val1']), export_id_hash(match['val2'])) for _,match in matches.items()]

    index_dict = lambda x: dict((v.item(),k) for k,v in enumerate(x))
    orig_face_map = index_dict(orig_brep.face_export_ids)
    orig_edge_map = index_dict(orig_brep.edge_export_ids)
    orig_vert_map = index_dict(orig_brep.vertex_export_ids)

    var_face_map = index_dict(var_brep.face_export_ids)
    var_edge_map = index_dict(var_brep.edge_export_ids)
    var_vert_map = index_dict(var_brep.vertex_export_ids)

    face_matches = []
    edge_matches = []
    vert_matches = []
    for orig_id, var_id in hashed_matches:
        if orig_id in orig_face_map:
            assert(var_id in var_face_map)
            face_matches.append([orig_face_map[orig_id], var_face_map[var_id]])
        elif orig_id in orig_edge_map:
            assert(var_id in var_edge_map)
            edge_matches.append([orig_edge_map[orig_id], var_edge_map[var_id]])
        elif orig_id in orig_vert_map:
            assert(var_id in var_vert_map)
            vert_matches.append([orig_vert_map[orig_id], var_vert_map[var_id]])
        else:
            pass # The last match in our test wasn't in the dataset and was JFD, JFD -- special case?
            #assert(False) # Error - missing export id

    face_matches = torch.tensor(face_matches).long().T if len(face_matches) > 0 else torch.empty((2,0)).long()
    edge_matches = torch.tensor(edge_matches).long().T if len(edge_matches) > 0 else torch.empty((2,0)).long()
    vert_matches = torch.tensor(vert_matches).long().T if len(vert_matches) > 0 else torch.empty((2,0)).long()

    data = zip_hetdata(orig_brep, var_brep)
    data.faces_matches = face_matches
    data.__edge_sets__['faces_matches'] = ['left_faces', 'right_faces']
    data.edges_matches = edge_matches
    data.__edge_sets__['edges_matches'] = ['left_edges', 'right_edges']
    data.vertices_matches = vert_matches
    data.__edge_sets__['vertices_matches'] = ['left_vertices', 'right_vertices']

    return data

follow_batch=['left_vertices','right_vertices','left_edges', 'right_edges','left_faces','right_faces', 'faces_matches', 'edges_matches', 'vertices_matches']
class BRepMatchingDataset(torch.utils.data.Dataset):
    def __init__(self, zip_path=None, cache_path=None, debug=False, mode='train', seed=42, test_size=0.1, val_size=0.1, test_identity=False):
        self.debug = debug
        do_preprocess = True
        if cache_path is not None:
            if os.path.exists(cache_path):
                cached_data = torch.load(cache_path)
                self.preprocessed_data = cached_data['preprocessed_data']
                self.group = cached_data['group']
                do_preprocess = False

        if do_preprocess:
            assert(zip_path is not None) # Must provide a zip path if cache does not exist
            self.preprocessed_data = []
            with ZipFile(zip_path, 'r') as zf:
                with zf.open('data/VariationData/all_variations.csv','r') as f:
                    variations = pd.read_csv(f)
                orig_id_dict = dict((k,v) for v,k in enumerate(variations.ps_orig.unique()))
                self.group = []
                for i in tqdm(range(len(variations))):
                    variation_record = variations.iloc[i]

                    m_path = 'data/Matches/' + variation_record.matchFile
                    o_path = 'data/BrepsWithReference/' + variation_record.ps_orig
                    v_path = 'data/BrepsWithReference/' + variation_record.ps_var

                    data = make_match_data(zf, o_path, v_path, m_path)
                    if data is not None:
                        self.group.append(orig_id_dict[variation_record.ps_orig])
                        self.preprocessed_data.append(data)
            self.group = torch.tensor(self.group).long()
            if cache_path is not None:
                os.makedirs(os.path.dirname(cache_path),exist_ok=True)
                cached_data = {
                    'preprocessed_data':self.preprocessed_data,
                    'group':self.group
                }
                torch.save(cached_data, cache_path)
        
        self.mode = mode
        unique_groups = self.group.unique()
        n_test = int(len(unique_groups)*test_size) if test_size < 1 else test_size
        n_val = int(len(unique_groups)*val_size) if val_size < 1 else val_size
        train_groups, test_groups = train_test_split(unique_groups, test_size=n_test, random_state=seed)
        train_groups, val_groups = train_test_split(train_groups, test_size=n_val, random_state=seed)
        groups_to_use = set((train_groups if self.mode == 'train' else test_groups if self.mode == 'test' else val_groups).tolist())

        self.preprocessed_data = [self.preprocessed_data[i] for i,g in enumerate(self.group) if g.item() in groups_to_use]

        self.test_identity = test_identity
        
    def __getitem__(self, idx):
        data = self.preprocessed_data[idx]
        if self.debug:
            data = self.preprocessed_data[idx]
            assert len(data.faces_matches[0].unique()) == len(data.faces_matches[0])
            assert len(data.faces_matches[1].unique()) == len(data.faces_matches[1])
        if self.test_identity:
            n_faces = data.left_faces.shape[0]
            n_edges = data.left_edges.shape[0]
            n_verts = data.left_vertices.shape[0]
            face_matches = torch.stack([torch.arange(n_faces).long(), torch.arange(n_faces).long()]) if n_faces > 0 else torch.empty((2,0)).long()
            edge_matches = torch.stack([torch.arange(n_edges).long(), torch.arange(n_edges).long()]) if n_edges > 0 else torch.empty((2,0)).long()
            vert_matches = torch.stack([torch.arange(n_verts).long(), torch.arange(n_verts).long()]) if n_verts > 0 else torch.empty((2,0)).long()

            for k in data.keys:
                if k.startswith('left'):
                    data[f'right{k[4:]}'] = data[k]

            data['faces_matches'] = face_matches
            data['edges_matches'] = edge_matches
            data['vertices_matches'] = vert_matches
        return data
    
    def __len__(self):
        return len(self.preprocessed_data)



class BRepMatchingDataModule(pl.LightningDataModule):
    def __init__(self, 
    batch_size: int = 16, 
    num_workers: int = 10, 
    persistent_workers: bool = True,
    shuffle: bool = True, 
    zip_path: str = None, 
    cache_path: str = None, 
    debug_data: bool = False,
    seed: int = 42,
    test_size: float = 0.1,
    val_size: float = 0.1,
    single_set: bool = False,
    test_identity: bool = False
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.shuffle = shuffle
        self.zip_path = zip_path
        self.cache_path = cache_path
        self.debug = debug_data
        self.seed = seed
        self.test_size = test_size
        self.val_size = val_size
        self.single_set = single_set
        self.test_identity = test_identity

        self.prepare_data_per_node = False #workaround to seeming bug in lightning

    def setup(self, **kwargs):
        super().__init__()
        self.train_ds = BRepMatchingDataset(zip_path=self.zip_path, cache_path=self.cache_path, debug=self.debug, seed = self.seed, test_size = self.test_size, val_size = self.val_size, mode='train', test_identity=self.test_identity)
        if self.single_set:
            self.test_ds = self.train_ds
            self.val_ds = self.train_ds
        else:
            self.test_ds = BRepMatchingDataset(zip_path=self.zip_path, cache_path=self.cache_path, debug=self.debug, seed = self.seed, test_size = self.test_size, val_size = self.val_size, mode='test', test_identity=self.test_identity)
            self.val_ds = BRepMatchingDataset(zip_path=self.zip_path, cache_path=self.cache_path, debug=self.debug, seed = self.seed, test_size = self.test_size, val_size = self.val_size, mode='val', test_identity=self.test_identity)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=self.shuffle, persistent_workers=self.persistent_workers, follow_batch=follow_batch)

    def val_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=False, persistent_workers=self.persistent_workers, follow_batch=follow_batch)

    def test_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=False, persistent_workers=self.persistent_workers, follow_batch=follow_batch)