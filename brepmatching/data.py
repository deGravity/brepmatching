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

from .utils import zip_hetdata, make_containing_dir, fix_file_descriptors
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from coincidence_matching import match_parts, match_parts_dict, get_export_id_types
from .transforms import *
import warnings

def normalize_pair(
    orig_part: Part,
    orig_part_data: str,
    var_part: Part,
    var_part_data: str,
    options: PartFeatures
):
    orig_bb = orig_part.summary.bounding_box
    var_bb = var_part.summary.bounding_box

    bb_corners = np.concatenate([orig_bb, var_bb],axis=0)

    joint_min = bb_corners.min(axis=0)
    joint_max = bb_corners.max(axis=0)
    diag = joint_max - joint_min
    center = (joint_min + joint_max) / 2
    scale = diag.max() / 2

    xfrm = np.array([
        [1, 0, 0, -center[0]],
        [0, 1, 0, -center[1]],
        [0, 0, 1, -center[2]],
        [0, 0, 0,      scale]
    ])

    options.transform = True
    options.transform_matrix = xfrm

    orig_part_xfrmed = Part(orig_part_data, options)
    var_part_xfrmed = Part(var_part_data, options)

    if not (orig_part_xfrmed.is_valid and var_part_xfrmed.is_valid):
        return None, None, 'bad_transform'

    try:
        conservative_transform_orig = (
            (orig_part.brep.relations.edge_to_vertex == orig_part_xfrmed.brep.relations.edge_to_vertex).all() and 
            (orig_part.brep.relations.face_to_face == orig_part_xfrmed.brep.relations.face_to_face).all() and 
            (orig_part.brep.relations.face_to_loop == orig_part_xfrmed.brep.relations.face_to_loop).all() and 
            (orig_part.brep.relations.loop_to_edge == orig_part_xfrmed.brep.relations.loop_to_edge).all()).item()
        
        conservative_transform_var = (
            (var_part.brep.relations.edge_to_vertex == var_part_xfrmed.brep.relations.edge_to_vertex).all() and 
            (var_part.brep.relations.face_to_face   == var_part_xfrmed.brep.relations.face_to_face).all() and 
            (var_part.brep.relations.face_to_loop   == var_part_xfrmed.brep.relations.face_to_loop).all() and 
            (var_part.brep.relations.loop_to_edge   == var_part_xfrmed.brep.relations.loop_to_edge).all()).item()
        
        if not (conservative_transform_orig and conservative_transform_var):
            return None, None, 'topo_change_xfrm'

    except Exception as e:
        return None, None, 'topo_change_xfrm'

    return orig_part_xfrmed, var_part_xfrmed, 'ok'




def make_match_data(
    zf, 
    orig_path, 
    var_path, 
    match_path, 
    bl_o_path, 
    bl_v_path, 
    bl_m_path, 
    include_meshes=True, 
    skip_onshape_baseline=False, 
    return_reason = False, 
    normalize=False, 
    use_mesh_quality=False, 
    mesh_quality=0.01
):
    has_baseline_data = False
    if orig_path not in zf.namelist() or var_path not in zf.namelist() or match_path not in zf.namelist():
        if return_reason:
            return None, 'missing_path'
        return None
    if bl_o_path is not None:
        has_baseline_data = True
        if bl_o_path not in zf.namelist() or bl_v_path not in zf.namelist() or bl_m_path not in zf.namelist():
            has_baseline_data = False
    options = PartOptions()
    if use_mesh_quality:
        options.set_quality = True
        options.quality = mesh_quality
    if not include_meshes:
        options.tesselate = False
    with zf.open(match_path,'r') as f:
        matches = json.load(f)
    with zf.open(orig_path, 'r') as f:
        orig_part_data = f.read().decode('utf-8')
        orig_part = Part(orig_part_data, options)
    if not orig_part.is_valid:
        if return_reason:
            return None, 'orig_invalid'
        return None
    with zf.open(var_path, 'r') as f:
        var_part_data = f.read().decode('utf-8')
        var_part = Part(var_part_data, options)
    if not var_part.is_valid:
        if return_reason:
            return None, 'var_invalid'
        return None
    if normalize:
        orig_part, var_part, reason = normalize_pair(orig_part, orig_part_data, var_part, var_part_data, options)
        if reason != 'ok':
            return None, reason
    features = PartFeatures()
    if not include_meshes:
        features.mesh = False
    orig_brep = part_to_graph(orig_part, features)
    var_brep = part_to_graph(var_part, features)
    for brep in (orig_brep, var_brep):
        if any([torch.any(torch.isnan(feat)) for feat in (brep.faces, brep.edges, brep.vertices)]):
            if return_reason:
                return None, 'has_nans'
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

    orig_id_types = get_export_id_types(orig_part_data)

    # Setup Ground Truth Matches
    face_matches, edge_matches, vert_matches = make_match_tensors(
        matches, 
        export_id_hash, 
        match2tensor, 
        orig_face_map, 
        orig_edge_map, 
        orig_vert_map, 
        var_face_map, 
        var_edge_map, 
        var_vert_map,
        orig_id_types)

    data = zip_hetdata(orig_brep, var_brep)
    data.faces_matches = face_matches
    data.__edge_sets__['faces_matches'] = ['left_faces', 'right_faces']
    data.edges_matches = edge_matches
    data.__edge_sets__['edges_matches'] = ['left_edges', 'right_edges']
    data.vertices_matches = vert_matches
    data.__edge_sets__['vertices_matches'] = ['left_vertices', 'right_vertices']

    # Setup Baseline Matching
    baseline_matching = match_parts(orig_part_data, var_part_data, False)
    bl_exact_face_matches = []
    bl_exact_edge_matches = []
    bl_exact_vert_matches = []
    bl_overlap_face_matches = []
    bl_overlap_edge_matches = []
    for matchings, orig_map, var_map, out_list in [
                (baseline_matching.face_matches, orig_face_map, var_face_map, bl_exact_face_matches),
                (baseline_matching.edge_matches, orig_edge_map, var_edge_map, bl_exact_edge_matches),
                (baseline_matching.vertex_matches, orig_vert_map, var_vert_map, bl_exact_vert_matches),
                (baseline_matching.face_overlaps, orig_face_map, var_face_map, bl_overlap_face_matches),
                (baseline_matching.edge_overlaps, orig_edge_map, var_edge_map, bl_overlap_edge_matches)
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
    bl_overlap_face_matches = match2tensor(bl_overlap_face_matches)
    bl_overlap_edge_matches = match2tensor(bl_overlap_edge_matches)


    data.bl_exact_faces_matches = bl_exact_face_matches
    data.__edge_sets__['bl_exact_faces_matches'] = ['left_faces', 'right_faces']
    data.bl_exact_edges_matches = bl_exact_edge_matches
    data.__edge_sets__['bl_exact_edges_matches'] = ['left_edges', 'right_edges']
    data.bl_exact_vertices_matches = bl_exact_vert_matches
    data.__edge_sets__['bl_exact_vertices_matches'] = ['left_vertices', 'right_vertices']
    data.bl_overlap_faces_matches = bl_overlap_face_matches
    data.__edge_sets__['bl_overlap_faces_matches'] = ['left_faces', 'right_faces']
    data.bl_overlap_edges_matches = bl_overlap_edge_matches
    data.__edge_sets__['bl_overlap_edges_matches'] = ['left_edges', 'right_edges']

    data.bl_overlap_larger_face_percentages = torch.tensor(baseline_matching.larger_face_overlap_percentages)
    data.__node_sets__.add('bl_overlap_larger_face_percentages')
    data.bl_overlap_smaller_face_percentages = torch.tensor(baseline_matching.smaller_face_overlap_percentages)
    data.__node_sets__.add('bl_overlap_smaller_face_percentages')
    data.bl_overlap_larger_edge_percentages = torch.tensor(baseline_matching.larger_edge_overlap_percentages)
    data.__node_sets__.add('bl_overlap_larger_edge_percentages')
    data.bl_overlap_smaller_edge_percentages = torch.tensor(baseline_matching.smaller_edge_overlap_percentages)
    data.__node_sets__.add('bl_overlap_smaller_edge_percentages')

    # Setup Onshape Baseline
    if bl_m_path is None or bl_o_path is None or bl_v_path is None or not has_baseline_data or skip_onshape_baseline:
        
        data.os_bl_faces_matches = torch.empty((2,0)).long()
        data.__edge_sets__['os_bl_faces_matches'] = ['left_faces', 'right_faces']
        data.os_bl_edges_matches = torch.empty((2,0)).long()
        data.__edge_sets__['os_bl_edges_matches'] = ['left_edges', 'right_edges']
        data.os_bl_vertices_matches = torch.empty((2,0)).long()
        data.__edge_sets__['os_bl_vertices_matches'] = ['left_vertices', 'right_vertices']
        data.n_onshape_baseline_unmatched = torch.tensor([0]).long()
        data.has_onshape_baseline = torch.tensor([False]).bool()
        if return_reason:
            return data, 'ok'
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
    var_types = get_export_id_types(bl_var_part_data)
    orig_var_types = get_export_id_types(var_part_data)
    num_onshape_baseline_unmatched = 0
    renamed_matches = {}
    for k,match in bl_matches.items():
        orig_export_id = match['val1']
        bl_var_export_id = match['val2']
        if bl_var_export_id in var_rename:
            new_export_id = var_rename[bl_var_export_id]
            assert(new_export_id in orig_var_types) # We've renamed the topo to something actually in the part
            assert(var_types[bl_var_export_id] == orig_id_types[orig_export_id]) # Make sure the types match up
            renamed_matches[k] = {'val1':orig_export_id, 'val2':new_export_id}
        elif var_types[bl_var_export_id] in ['PK_CLASS_face', 'PK_CLASS_edge', 'PK_CLASS_vertex']: # Should we have matched this but didn't?
            num_onshape_baseline_unmatched += 1
    
    os_bl_face_matches, os_bl_edge_matches, os_bl_vert_matches = make_match_tensors(
        renamed_matches, 
        export_id_hash, 
        match2tensor, 
        orig_face_map, 
        orig_edge_map, 
        orig_vert_map, 
        var_face_map, 
        var_edge_map, 
        var_vert_map,
        orig_id_types)
    
    data.os_bl_faces_matches = os_bl_face_matches
    data.__edge_sets__['os_bl_faces_matches'] = ['left_faces', 'right_faces']
    data.os_bl_edges_matches = os_bl_edge_matches
    data.__edge_sets__['os_bl_edges_matches'] = ['left_edges', 'right_edges']
    data.os_bl_vertices_matches = os_bl_vert_matches
    data.__edge_sets__['os_bl_vertices_matches'] = ['left_vertices', 'right_vertices']
    data.n_onshape_baseline_unmatched = torch.tensor([num_onshape_baseline_unmatched]).long()
    data.has_onshape_baseline = torch.tensor([True]).bool()

    if return_reason:
        return data, 'ok'
    return data

def make_match_tensors(matches, export_id_hash, match2tensor, orig_face_map, orig_edge_map, orig_vert_map, var_face_map, var_edge_map, var_vert_map, orig_classes):
    face_matches = []
    edge_matches = []
    vert_matches = []
    hashed_matches = [(export_id_hash(match['val1']), export_id_hash(match['val2'])) for _,match in matches.items()]
    hashed_orig_classes = {export_id_hash(k):v for k,v in orig_classes.items()}

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
            if hashed_orig_classes[orig_id] in ['PK_CLASS_face', 'PK_CLASS_vertex','PK_CLASS_edge']:
                assert(False)

    face_matches = match2tensor(face_matches)
    edge_matches = match2tensor(edge_matches)
    vert_matches = match2tensor(vert_matches)
    return face_matches,edge_matches,vert_matches

def identity_collate(data):
    return data[0]

class ParallelPreprocessor(torch.utils.data.Dataset):

    @classmethod
    def preprocess(cls, 
        zip_path, 
        cache_path, 
        num_workers=8,
        use_hq=False,
        quality=0.01,
        normalize=False
    ):
        fix_file_descriptors()
        warnings.filterwarnings('ignore', category=UserWarning) # hide annoying warnings
        
        processor = ParallelPreprocessor(
            zip_path, cache_path,
            use_hq=use_hq,
            quality=quality,
            normalize=normalize)

        loader = torch.utils.data.DataLoader(
            processor, 
            batch_size=1, 
            num_workers=num_workers, 
            shuffle=False, 
            collate_fn = identity_collate
            )
        
        dataset = []
        for d in tqdm(loader): # Turn of multiprocessing for now since it was crashing
            dataset.append(d)

        torch.save(dataset, cache_path)

        return dataset


    def __init__(self, zip_path, cache_path, use_hq=False,
        quality=0.01,
        normalize=False):
        with ZipFile(zip_path, 'r') as zf:
            with zf.open('data/VariationData/all_variations.csv','r') as f:
                self.variations = pd.read_csv(f)
            if 'data/baseline/allVariationsWithBaseline.csv' in zf.namelist():
                with zf.open('data/baseline/allVariationsWithBaseline.csv','r') as f:
                    self.variations = pd.read_csv(f)
            
        self.orig_id_dict = dict((k,v) for v,k in enumerate(self.variations.ps_orig.unique()))


        self.n = len(self.variations)
        self.zf = None
        self.zip_path = zip_path
        self.cache_path = cache_path

        self.use_hq=use_hq
        self.quality=quality
        self.normalize=normalize

    def get_zip(self):
        if self.zf is None:
            self.zf = ZipFile(self.zip_path, 'r')
        return self.zf
    
    def __del__(self):
        try:
            if self.zf is not None:
                self.zf.close()
        finally:
            self.zf = None

    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        return self.process(idx)

    def process(self, idx):
        variation_record = self.variations.iloc[idx]
        variation_index = self.variations.index[idx]

        if 'fail' in self.variations.columns and variation_record.fail == 1:
            return {
            'idx':idx,
            'group': self.orig_id_dict[variation_record.ps_orig],
            'data': None,
            'original_index': variation_index,
            'reason': 'fail'
        }

        m_path = 'data/Matches/' + variation_record.matchFile
        o_path = 'data/BrepsWithReference/' + variation_record.ps_orig
        v_path = 'data/BrepsWithReference/' + variation_record.ps_var

        bl_m_path = None
        bl_o_path = None
        bl_v_path = None

        skip_onshape_baseline = True

        if 'baselineMatch' in self.variations.columns:
            bl_m_path = 'data/baseline/' + variation_record.baselineMatch
            bl_o_path = 'data/baseline/' + variation_record.baselineOrig
            bl_v_path = 'data/baseline/' + variation_record.baselineNew

            skip_onshape_baseline = variation_record.translationFail == 1

        zf = self.get_zip()

        data, reason = make_match_data(
            zf, o_path, v_path, m_path, bl_o_path, 
            bl_v_path, bl_m_path, 
            skip_onshape_baseline=skip_onshape_baseline, 
            return_reason=True,
            normalize=self.normalize, 
            use_mesh_quality=self.use_hq, 
            mesh_quality=self.quality)

        return {
            'idx': idx,
            'group': self.orig_id_dict[variation_record.ps_orig],
            'data': data,
            'original_index': variation_index,
            'reason': reason
        }

def load_data(zip_path=None, cache_path=None):
    if cache_path is not None:
        if os.path.exists(cache_path):
            return torch.load(cache_path)

    assert(zip_path is not None) # Must provide a zip path if cache does not exist
    warnings.filterwarnings('ignore', category=UserWarning) # hide annoying warnings
    preprocessed_data = []
    with ZipFile(zip_path, 'r') as zf:
        with zf.open('data/VariationData/all_variations.csv','r') as f:
            variations = pd.read_csv(f)
        if 'data/baseline/allVariationsWithBaseline.csv' in zf.namelist():
            with zf.open('data/baseline/allVariationsWithBaseline.csv','r') as f:
                variations = pd.read_csv(f)
        if 'fail' in variations.columns:
            variations = variations[variations.fail == 0]
        orig_id_dict = dict((k,v) for v,k in enumerate(variations.ps_orig.unique()))
        group = []
        original_index = []
        for i in tqdm(range(len(variations)), "Preprocessing Data"):
            variation_record = variations.iloc[i]
            variation_index = variations.index[i]

            m_path = 'data/Matches/' + variation_record.matchFile
            o_path = 'data/BrepsWithReference/' + variation_record.ps_orig
            v_path = 'data/BrepsWithReference/' + variation_record.ps_var

            bl_m_path = None
            bl_o_path = None
            bl_v_path = None

            skip_onshape_baseline = True

            if 'baselineMatch' in variations.columns:
                bl_m_path = 'data/baseline/' + variation_record.baselineMatch
                bl_o_path = 'data/baseline/' + variation_record.baselineOrig
                bl_v_path = 'data/baseline/' + variation_record.baselineNew

                skip_onshape_baseline = variation_record.translationFail == 1

            data = make_match_data(zf, o_path, v_path, m_path, bl_o_path, bl_v_path, bl_m_path, skip_onshape_baseline=skip_onshape_baseline)
            if data is not None:
                group.append(orig_id_dict[variation_record.ps_orig])
                preprocessed_data.append(data)
                original_index.append(variation_index)
    group = torch.tensor(group).long()
    cached_data = {
        'preprocessed_data':preprocessed_data,
        'group':group,
        'original_index':original_index
    }
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path),exist_ok=True)
        torch.save(cached_data, cache_path)
    return cached_data


follow_batch=['left_vertices','right_vertices','left_edges', 'right_edges','left_faces','right_faces', 'faces_matches', 'edges_matches', 'vertices_matches']
class BRepMatchingDataset(torch.utils.data.Dataset):
    def __init__(self, cached_data, debug=False, mode='train', seed=42, test_size=0.1, val_size=0.1, test_identity=False, transforms=None, require_onshape_matchings=True, enable_blacklist=True):
        self.debug = debug
        self.transforms = compose(*transforms[::-1]) if transforms else None

        self.preprocessed_data = cached_data['preprocessed_data']
        self.group = cached_data['group']
        self.original_index = cached_data['original_index']

        self.mode = mode
        unique_groups = self.group.unique()
        n_test = int(len(unique_groups)*test_size) if test_size < 1 else test_size
        n_val = int(len(unique_groups)*val_size) if val_size < 1 else val_size
        
        if val_size > 0:
            train_groups, val_groups = train_test_split(unique_groups, test_size=n_val, random_state=seed)
        else:
            train_groups = unique_groups
            val_groups = np.array([],dtype=int)
        if test_size > 0:
            train_groups, test_groups = train_test_split(train_groups, test_size=n_test, random_state=seed)
        else:
            test_groups = np.array([],dtype=int)
        
        groups_to_use = set((train_groups if self.mode == 'train' else test_groups if self.mode == 'test' else val_groups).tolist())

        self.preprocessed_data = [self.preprocessed_data[i] for i,g in enumerate(self.group) if g.item() in groups_to_use]

        self.test_identity = test_identity

        self.original_index = [self.original_index[i] for i,g in enumerate(self.group) if g.item() in groups_to_use]

        self.require_onshape_matchings = require_onshape_matchings
        if require_onshape_matchings:
            data_with_onshape_matchings = [i for i,d in enumerate(self.preprocessed_data) if hasattr(d, 'has_onshape_baseline') and d.has_onshape_baseline]
            self.preprocessed_data = [self.preprocessed_data[i] for i in data_with_onshape_matchings]
            self.original_index = [self.original_index[i] for i in data_with_onshape_matchings]

        if enable_blacklist:
            blacklist = {3096, 3097, 3098, 6372, 6373, 6374}
            self.preprocessed_data = [self.preprocessed_data[i] for i, o in enumerate(self.original_index) if o not in blacklist]
            self.original_index = [o for o in self.original_index if o not in blacklist]
        
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
    exact_match_labels: bool = False,
    val_batch_size: int = None,
    test_batch_size: int = None,
    enable_blacklist: bool = True,
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
        self.val_batch_size = batch_size if val_batch_size is None else val_batch_size
        self.test_batch_size = self.val_batch_size if test_batch_size is None else test_batch_size
        self.enable_blacklist = enable_blacklist

        print("train_batch_size", self.batch_size, "val_batch_size", self.val_batch_size, "test_batch_size", self.test_batch_size)

        self.prepare_data_per_node = False #workaround to seeming bug in lightning

    def setup(self, **kwargs):
        super().__init__()
        transforms = []
        if self.exact_match_labels:
            transforms.append(use_bl_exact_match_labels)
        cached_data = load_data(zip_path=self.zip_path, cache_path=self.cache_path)
        self.train_ds = BRepMatchingDataset(cached_data=cached_data, debug=self.debug, seed = self.seed, test_size = self.test_size, val_size = self.val_size, mode='train', test_identity=self.test_identity, transforms=transforms, enable_blacklist=self.enable_blacklist)
        if self.single_set:
            self.test_ds = self.train_ds
            self.val_ds = self.train_ds
        else:
            self.test_ds = BRepMatchingDataset(cached_data=cached_data, debug=self.debug, seed = self.seed, test_size = self.test_size, val_size = self.val_size, mode='test', test_identity=self.test_identity, transforms=transforms, enable_blacklist=self.enable_blacklist)
            self.val_ds = BRepMatchingDataset(cached_data=cached_data, debug=self.debug, seed = self.seed, test_size = self.test_size, val_size = self.val_size, mode='val', test_identity=self.test_identity, transforms=transforms, enable_blacklist=self.enable_blacklist)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=self.shuffle, persistent_workers=self.persistent_workers, follow_batch=follow_batch)

    def val_dataloader(self):
        # TODO: fix this thing
        return DataLoader(self.val_ds, batch_size=self.val_batch_size, num_workers=self.num_workers,shuffle=False, persistent_workers=self.persistent_workers, follow_batch=follow_batch)

    def test_dataloader(self):
        # TODO: fix this thing
        return DataLoader(self.test_ds, batch_size=self.test_batch_size, num_workers=self.num_workers,shuffle=False, persistent_workers=self.persistent_workers, follow_batch=follow_batch)


def make_filter(cache, face_thresh = 1000, edge_thresh = 1000, vert_thresh = 1000, ignore_origins = False):
    filt = []
    for d in cache:
        keep = False
        if d['reason'] == 'ok':
            keep = True
            data = d['data']
            has_baseline = data.has_onshape_baseline[0].item()
            if has_baseline:
                baseline_lost = data.n_onshape_baseline_unmatched[0].item()

                left_faces = data.left_faces.clone()
                right_faces = data.right_faces.clone()

                left_edges = data.left_edges.clone()
                right_edges = data.right_edges.clone()


                if ignore_origins: # columns 15-18 are the origin parameters. Sometimes these are exactly +/- 500, which is way out of distribution
                    left_faces = torch.cat([left_faces[:,:15], left_faces[:,18:]],dim=1)
                    left_edges = torch.cat([left_edges[:,:15], left_edges[:,18:]],dim=1)

                    right_faces = torch.cat([right_faces[:,:15], right_faces[:,18:]],dim=1)
                    right_edges = torch.cat([right_edges[:,:15], right_edges[:,18:]],dim=1)                    

                face_max = max(
                        left_faces.abs().max().item() if len(left_faces) > 0 else 0., 
                        right_faces.abs().max().item() if len(right_faces) > 0 else 0.
                    )

                edge_max = max(
                        left_edges.abs().max().item() if len(left_edges) > 0 else 0.,
                        right_edges.abs().max().item()if len(right_edges) > 0 else 0.
                    )
                

                vert_max = max(
                        data.left_vertices.abs().max().item() if len(data.left_vertices) > 0 else 0., 
                        data.right_vertices.abs().max().item() if len(data.right_vertices) > 0 else 0.
                    )
                
                if baseline_lost > 0:
                    keep = False
                
                if face_max > face_thresh:
                    keep = False

                if edge_max > edge_thresh:
                    keep = False
                
                if vert_max > vert_thresh:
                    keep = False

            else:
                keep = False
        filt.append(keep)
    filt = np.array(filt)
    return filt

def convert_cache(cache, face_thresh = 1000, edge_thresh = 1000, vert_thresh = 1000, ignore_origins = False):

    preprocessed_data = []
    original_index = []
    group = []

    filt = make_filter(cache, face_thresh, edge_thresh, vert_thresh, ignore_origins)
    for f,d in zip(filt, cache):
        if f:
            original_index.append(d['original_index'])
            group.append(d['group'])
            preprocessed_data.append(d['data'])
            pass

    group = torch.tensor(group).long()
    cached_data = {
        'preprocessed_data':preprocessed_data,
        'group':group,
        'original_index':original_index
    }

    return cached_data


def convert_cache_pt(in_path, out_path):
    cache = torch.load(in_path)
    converted = convert_cache(cache)
    make_containing_dir(out_path)
    torch.save(converted, out_path)
