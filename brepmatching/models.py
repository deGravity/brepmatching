import torch
from automate import SBGCN

from .utils import zip_apply


class PairEmbedder(torch.nn.Module):
    def __init__(self,
        s_face = 62,
        s_loop = 38,
        s_edge = 72,
        s_vert = 3,
        embedding_size = 64,
        k = 6,
    ):
        super().__init__()
        self.sbgcn = SBGCN(s_face, s_loop, s_edge, s_vert, embedding_size, k)
    
    def forward(self, batch):
        orig_embeddings, var_embeddings = zip_apply(self.sbgcn, batch)
        _, _, f_orig, _, e_orig, v_orig = orig_embeddings
        _, _, f_var, _, e_var, v_var = var_embeddings
        return (f_orig, e_orig, v_orig), (f_var, e_var, v_var)