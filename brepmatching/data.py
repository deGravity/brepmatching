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

from coincidence_matching import match_parts, match_parts_dict
from .transforms import *



def make_match_data(zf, orig_path, var_path, match_path, bl_o_path, bl_v_path, bl_m_path, include_meshes=True):
    if orig_path not in zf.namelist() or var_path not in zf.namelist() or match_path not in zf.namelist():
        return None
    if bl_o_path is not None:
        if bl_o_path not in zf.namelist() or bl_v_path not in zf.namelist() or bl_m_path not in zf.namelist():
            return None
    options = PartOptions()
    if not include_meshes:
        options.tesselate = False
    with zf.open(match_path,'r') as f:
        matches = json.load(f)
    with zf.open(orig_path, 'r') as f:
        orig_part_data = f.read().decode('utf-8')
        orig_part = Part(orig_part_data, options)
    if not orig_part.is_valid:
        return None
    with zf.open(var_path, 'r') as f:
        var_part_data = f.read().decode('utf-8')
        var_part = Part(var_part_data, options)
    if not var_part.is_valid:
        return None
    features = PartFeatures()
    if not include_meshes:
        features.mesh = False
    orig_brep = part_to_graph(orig_part, features)
    var_brep = part_to_graph(var_part, features)
    for brep in (orig_brep, var_brep):
        if any([torch.any(torch.isnan(feat)) for feat in (brep.faces, brep.edges, brep.vertices)]):
            return None

    export_id_hash = lambda x: xxhash.xxh32(x).intdigest()
    match2tensor = lambda x: torch.tensor(x).long().T if len(x) > 0 else torch.empty((2,0)).long()
    

    index_dict = lambda x: dict((v.item(),k) for k,v in enumerate(x))
    orig_face_map = index_dict(orig_brep.face_export_ids)
    orig_edge_map = index_dict(orig_brep.edge_export_ids)
    orig_vert_map = index_dict(orig_brep.vertex_export_ids)

    var_face_map = index_dict(var_brep.face_export_ids)
    var_edge_map = index_dict(var_brep.edge_export_ids)
    var_vert_map = index_dict(var_brep.vertex_export_ids)

    # Setup Ground Truth Matches
    face_matches, edge_matches, vert_matches = make_match_tensors(matches, export_id_hash, match2tensor, orig_face_map, orig_edge_map, orig_vert_map, var_face_map, var_edge_map, var_vert_map)

    data = zip_hetdata(orig_brep, var_brep)
    data.faces_matches = face_matches
    data.__edge_sets__['faces_matches'] = ['left_faces', 'right_faces']
    data.edges_matches = edge_matches
    data.__edge_sets__['edges_matches'] = ['left_edges', 'right_edges']
    data.vertices_matches = vert_matches
    data.__edge_sets__['vertices_matches'] = ['left_vertices', 'right_vertices']

    # Setup Baseline Matching
    baseline_matching = match_parts(orig_part_data, var_part_data, True) # TODO - set exact to false once implemented
    bl_exact_face_matches = []
    bl_exact_edge_matches = []
    bl_exact_vert_matches = []
    for matchings, orig_map, var_map, out_list in [
                (baseline_matching.face_matches, orig_face_map, var_face_map, bl_exact_face_matches),
                (baseline_matching.edge_matches, orig_edge_map, var_edge_map, bl_exact_edge_matches),
                (baseline_matching.vertex_matches, orig_vert_map, var_vert_map, bl_exact_vert_matches)
            ]:
        for o_t, v_t in matchings:
            o_t_h = export_id_hash(o_t)
            v_t_h = export_id_hash(v_t)
            assert(o_t_h in orig_map)
            assert(v_t_h in var_map)
            out_list.append([orig_map[o_t_h], var_map[v_t_h]])

    bl_exact_face_matches = match2tensor(bl_exact_face_matches)
    bl_exact_edge_matches = match2tensor(bl_exact_edge_matches)
    bl_exact_vert_matches = match2tensor(bl_exact_vert_matches)

    data.bl_exact_faces_matches = bl_exact_face_matches
    data.__edge_sets__['bl_exact_faces_matches'] = ['left_faces', 'right_faces']
    data.bl_exact_edges_matches = bl_exact_edge_matches
    data.__edge_sets__['bl_exact_edges_matches'] = ['left_edges', 'right_edges']
    data.bl_exact_vertices_matches = bl_exact_vert_matches
    data.__edge_sets__['bl_exact_vertices_matches'] = ['left_vertices', 'right_vertices']

    # Setup Onshape Baseline
    if bl_m_path is None or bl_o_path is None or bl_v_path is None:
        return data # Stop now if we don't have baselines in the dataset

    with zf.open(bl_m_path,'r') as f:
        bl_matches = json.load(f)
    with zf.open(bl_v_path, 'r') as f:
        bl_var_part_data = f.read().decode('utf-8')
    # The Onshape baseline variation may have different export ids than
    # the original variation file (since it does not have the construction
    # history to base the ids on). Re-map these back to the original variation
    # that the ground-truth matching is based on by running exact matching
    # (the two parts should match perfectly).
    var_rename = match_parts_dict(bl_var_part_data, var_part_data, True)
    missing_count = 0
    for k in bl_matches:
        if bl_matches[k]['val2'] in var_rename:
            bl_matches[k]['val2'] = var_rename[bl_matches[k]['val2']]
        else:
            missing_count += 1
    if missing_count > 0:
        print(f'Missing Matches: {missing_count}')
    os_bl_face_matches, os_bl_edge_matches, os_bl_vert_matches = make_match_tensors(
        bl_matches, 
        export_id_hash, 
        match2tensor, 
        orig_face_map, 
        orig_edge_map, 
        orig_vert_map, 
        var_face_map, 
        var_edge_map, 
        var_vert_map)
    
    data.os_bl_faces_matches = os_bl_face_matches
    data.__edge_sets__['os_bl_faces_matches'] = ['left_faces', 'right_faces']
    data.os_bl_edges_matches = os_bl_edge_matches
    data.__edge_sets__['os_bl_edges_matches'] = ['left_edges', 'right_edges']
    data.os_bl_vertices_matches = os_bl_vert_matches
    data.__edge_sets__['os_bl_vertices_matches'] = ['left_vertices', 'right_vertices']

    return data

def make_match_tensors(matches, export_id_hash, match2tensor, orig_face_map, orig_edge_map, orig_vert_map, var_face_map, var_edge_map, var_vert_map):
    face_matches = []
    edge_matches = []
    vert_matches = []
    hashed_matches = [(export_id_hash(match['val1']), export_id_hash(match['val2'])) for _,match in matches.items()]
    missing_faces = 0
    missing_edges = 0
    missing_verts = 0
    missing_orig = 0
    for orig_id, var_id in hashed_matches:
        if orig_id in orig_face_map:
            #assert(var_id in var_face_map)
            if var_id in var_face_map:
                face_matches.append([orig_face_map[orig_id], var_face_map[var_id]])
            else:
                missing_faces += 1
        elif orig_id in orig_edge_map:
            #assert(var_id in var_edge_map)
            if var_id in var_edge_map:
                edge_matches.append([orig_edge_map[orig_id], var_edge_map[var_id]])
            else:
                missing_edges += 1
        elif orig_id in orig_vert_map:
            #assert(var_id in var_vert_map)
            if var_id in var_vert_map:
                vert_matches.append([orig_vert_map[orig_id], var_vert_map[var_id]])
            else:
                missing_verts +=1
        else:
            missing_orig += 1
            pass # The last match in our test wasn't in the dataset and was JFD, JFD -- special case?
            #assert(False) # Error - missing export id
    if missing_faces + missing_edges + missing_verts > 0:
        print(f'Missing: {missing_faces} faces, {missing_edges} edges, {missing_verts} verts, {missing_orig} origs')
    
    face_matches = match2tensor(face_matches)
    edge_matches = match2tensor(edge_matches)
    vert_matches = match2tensor(vert_matches)
    return face_matches,edge_matches,vert_matches

follow_batch=['left_vertices','right_vertices','left_edges', 'right_edges','left_faces','right_faces', 'faces_matches', 'edges_matches', 'vertices_matches']
class BRepMatchingDataset(torch.utils.data.Dataset):
    def __init__(self, zip_path=None, cache_path=None, debug=False, mode='train', seed=42, test_size=0.1, val_size=0.1, test_identity=False, transforms=None):
        self.debug = debug
        self.transforms = compose(*transforms[::-1]) if transforms else None
        do_preprocess = True
        if cache_path is not None:
            if os.path.exists(cache_path):
                cached_data = torch.load(cache_path)
                self.preprocessed_data = cached_data['preprocessed_data']
                self.group = cached_data['group']
                self.original_index = cached_data['original_index']
                do_preprocess = False

        if do_preprocess:
            assert(zip_path is not None) # Must provide a zip path if cache does not exist
            self.preprocessed_data = []
            with ZipFile(zip_path, 'r') as zf:
                with zf.open('data/VariationData/all_variations.csv','r') as f:
                    variations = pd.read_csv(f)
                if 'data/baseline/allVariationsWithBaseline.csv' in zf.namelist():
                    with zf.open('data/baseline/allVariationsWithBaseline.csv','r') as f:
                        variations = pd.read_csv(f)
                if 'fail' in variations.columns:
                    variations = variations[variations.fail == 0]
                orig_id_dict = dict((k,v) for v,k in enumerate(variations.ps_orig.unique()))
                self.group = []
                self.original_index = []
                for i in tqdm(range(len(variations)), "Preprocessing Data"):
                    variation_record = variations.iloc[i]
                    variation_index = variations.index[i]

                    m_path = 'data/Matches/' + variation_record.matchFile
                    o_path = 'data/BrepsWithReference/' + variation_record.ps_orig
                    v_path = 'data/BrepsWithReference/' + variation_record.ps_var

                    bl_m_path = None
                    bl_o_path = None
                    bl_v_path = None

                    if 'baselineMatch' in variations.columns:
                        bl_m_path = 'data/baseline/' + variation_record.baselineMatch
                        bl_o_path = 'data/baseline/' + variation_record.baselineOrig
                        bl_v_path = 'data/baseline/' + variation_record.baselineNew
                    data = make_match_data(zf, o_path, v_path, m_path, bl_o_path, bl_v_path, bl_m_path)
                    if data is not None:
                        self.group.append(orig_id_dict[variation_record.ps_orig])
                        self.preprocessed_data.append(data)
                        self.original_index.append(variation_index)
            self.group = torch.tensor(self.group).long()
            if cache_path is not None:
                os.makedirs(os.path.dirname(cache_path),exist_ok=True)
                cached_data = {
                    'preprocessed_data':self.preprocessed_data,
                    'group':self.group,
                    'original_index':self.original_index
                }
                torch.save(cached_data, cache_path)
        
        self.mode = mode
        unique_groups = self.group.unique()
        n_test = int(len(unique_groups)*test_size) if test_size < 1 else test_size
        n_val = int(len(unique_groups)*val_size) if val_size < 1 else val_size
        if test_size > 0:
            train_groups, test_groups = train_test_split(unique_groups, test_size=n_test, random_state=seed)
        else:
            train_groups = unique_groups
            test_groups = np.array([],dtype=int)
        if val_size > 0:
            train_groups, val_groups = train_test_split(train_groups, test_size=n_val, random_state=seed)
        else:
            val_groups = np.array([],dtype=int)
        groups_to_use = set((train_groups if self.mode == 'train' else test_groups if self.mode == 'test' else val_groups).tolist())

        self.preprocessed_data = [self.preprocessed_data[i] for i,g in enumerate(self.group) if g.item() in groups_to_use]

        self.test_identity = test_identity

        self.original_index = [self.original_index[i] for i,g in enumerate(self.group) if g.item() in groups_to_use]
        
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
        if self.transforms is not None:
            data = self.transforms(data)
        return data
    
    def variation_index(self, idx):
        return self.original_index[idx]
        
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
    test_identity: bool = False,
    exact_match_labels: bool = False
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
        self.exact_match_labels = exact_match_labels

        self.prepare_data_per_node = False #workaround to seeming bug in lightning

    def setup(self, **kwargs):
        super().__init__()
        transforms = []
        if self.exact_match_labels:
            transforms.append(use_bl_exact_match_labels)
        self.train_ds = BRepMatchingDataset(zip_path=self.zip_path, cache_path=self.cache_path, debug=self.debug, seed = self.seed, test_size = self.test_size, val_size = self.val_size, mode='train', test_identity=self.test_identity, transforms=transforms)
        if self.single_set:
            self.test_ds = self.train_ds
            self.val_ds = self.train_ds
        else:
            self.test_ds = BRepMatchingDataset(zip_path=self.zip_path, cache_path=self.cache_path, debug=self.debug, seed = self.seed, test_size = self.test_size, val_size = self.val_size, mode='test', test_identity=self.test_identity, transforms=transforms)
            self.val_ds = BRepMatchingDataset(zip_path=self.zip_path, cache_path=self.cache_path, debug=self.debug, seed = self.seed, test_size = self.test_size, val_size = self.val_size, mode='val', test_identity=self.test_identity, transforms=transforms)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=self.shuffle, persistent_workers=self.persistent_workers, follow_batch=follow_batch)

    def val_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=False, persistent_workers=self.persistent_workers, follow_batch=follow_batch)

    def test_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=False, persistent_workers=self.persistent_workers, follow_batch=follow_batch)