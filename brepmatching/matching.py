import torch
import torch_scatter
from torch.nn import Linear, Sequential, ModuleList, BatchNorm1d, Dropout, LeakyReLU, ReLU, Module
import torch_geometric as tg
from automate import SBGCN
from brepmatching.models import PairEmbedder


def outermatch(v1, v2):
    """
    v1: m x k
    v2: n x k

    returns 2 x m x n x k tensor from combining each row of v1 with each row of v2 for matching
    """
    v1e = v1.expand((v2.shape, *v1.shape)).transpose(0, 1)
    v2e = v2.expand((v1.shape, *v2.shape))
    return torch.stack([v1e, v2e])


def simplepad(x, dim, value, size):
    newshape = list(x.shape)
    newshape[dim] = size
    xpad = torch.full(newshape, value)
    xpad[:x.shape[0], :x.shape[1], :x.shape[2]] = x
    return xpad


class Matcher(Module):
    def __init__(self,
    f_in_width=62,
        l_in_width=38,
        e_in_width=72,
        v_in_width=3,
        out_width=64,
        k=6,
        #use_uvnet_features=False,
        #crv_in_dim=[0, 1, 2, 3, 4, 5],
        #srf_in_dim=[0, 1, 2, 3, 4, 5, 8],
        #crv_emb_dim=64,
        #srf_emb_dim=64,
        ):
        super().__init__()
        
        self.pair_embedder = PairEmbedder(f_in_width, l_in_width, e_in_width, v_in_width, out_width, k)


    def forward(self, data):
        emb = self.pair_embedder(data)

