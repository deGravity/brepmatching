import torch
from automate import SBGCN
from torch.nn import BatchNorm1d

from .utils import zip_apply_2, zip_apply, zip_hetdata


class BrepNormalizer(torch.nn.Module):
    def __init__(self,
        s_face = 62,
        s_loop = 38,
        s_edge = 72,):
        super().__init__()
        self.faces_bn = BatchNorm1d(s_face)
        self.edges_bn = BatchNorm1d(s_edge)
        self.loop_bn = BatchNorm1d(s_loop)
    
    def forward(self, data):
        data.faces = self.faces_bn(data.faces)
        data.edges = self.edges_bn(data.edges)
        data.loops = self.loop_bn(data.loops)
        return data


class PairEmbedder(torch.nn.Module):
    def __init__(self,
        s_face = 62,
        s_loop = 38,
        s_edge = 72,
        s_vert = 3,
        embedding_size = 64,
        k = 6,
        batch_norm=False
    ):
        super().__init__()
        self.batch_norm = batch_norm
        if batch_norm:
            self.norm_left = BrepNormalizer(s_face, s_loop, s_edge)
            self.norm_right = BrepNormalizer(s_face, s_loop, s_edge)
        self.sbgcn = SBGCN(s_face, s_loop, s_edge, s_vert, embedding_size, k)
    
    def forward(self, batch):
        if self.batch_norm:
            #TODO: normalize jointly
            batch1, batch2 = zip_apply_2(self.norm_left, self.norm_right, batch)
            batch = zip_hetdata(batch1, batch2)
        orig_embeddings, var_embeddings = zip_apply(self.sbgcn, batch)
        _, _, f_orig, _, e_orig, v_orig = orig_embeddings
        _, _, f_var, _, e_var, v_var = var_embeddings
        return (f_orig, e_orig, v_orig), (f_var, e_var, v_var)