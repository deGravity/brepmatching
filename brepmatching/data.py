import json
import os
from zipfile import ZipFile

import pandas as pd
import torch
import xxhash
from automate import Part, PartOptions, PartFeatures, part_to_graph
from tqdm import tqdm

from .utils import zip_hetdata


def make_match_data(zf, orig_path, var_path, match_path, include_meshes=True):
    options = PartOptions()
    if not include_meshes:
        options.tesselate = False
    with zf.open(match_path,'r') as f:
        matches = json.load(f)
    with zf.open(orig_path, 'r') as f:
        orig_part = Part(f.read().decode('utf-8'), options)
    with zf.open(var_path, 'r') as f:
        var_part = Part(f.read().decode('utf-8'), options)
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
    data.face_matches = face_matches
    data.__edge_sets__['face_matches'] = ['left_faces', 'right_faces']
    data.edge_matches = edge_matches
    data.__edge_sets__['edge_matches'] = ['left_edges', 'right_edges']
    data.vertex_matches = vert_matches
    data.__edge_sets__['vertex_matches'] = ['left_vertices', 'right_vertices']

    return data

class BRepMatchingDataset(torch.utils.data.Dataset):
    def __init__(self, zip_path=None, cache_path=None, debug=False):
        self.debug = debug
        do_preprocess = True
        if cache_path is not None:
            if os.path.exists(cache_path):
                self.preprocessed_data = torch.load(cache_path)
                do_preprocess = False

        if do_preprocess:
            assert(zip_path is not None) # Must provide a zip path if cache does not exist
            self.preprocessed_data = []
            with ZipFile(zip_path, 'r') as zf:
                with zf.open('data/VariationData/all_variations.csv','r') as f:
                    variations = pd.read_csv(f)

                for i in tqdm(range(len(variations))):
                    variation_record = variations.iloc[i]

                    m_path = 'data/Matches/' + variation_record.matchFile
                    o_path = 'data/BrepsWithReference/' + variation_record.ps_orig
                    v_path = 'data/BrepsWithReference/' + variation_record.ps_var

                    data = make_match_data(zf, o_path, v_path, m_path)
                    self.preprocessed_data.append(data)
            if cache_path is not None:
                os.makedirs(os.path.dirname(cache_path),exist_ok=True)
                torch.save(self.preprocessed_data, cache_path)
    
    def __getitem__(self, idx):
        if self.debug:
            data = self.preprocessed_data[idx]
            assert len(data.face_matches[0].unique()) == len(data.face_matches[0])
            assert len(data.face_matches[1].unique()) == len(data.face_matches[1])
        return self.preprocessed_data[idx]
    
    def __len__(self):
        return len(self.preprocessed_data)
