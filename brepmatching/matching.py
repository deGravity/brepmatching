import torch
import torch_scatter
from torch.nn import Linear, Sequential, ModuleList, BatchNorm1d, Dropout, LeakyReLU, ReLU, Module
import torch_geometric as tg
from automate import SBGCN


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
    f_in_width,
        l_in_width,
        e_in_width,
        v_in_width,
        out_width,
        k,
        use_uvnet_features=False,
        crv_in_dim=[0, 1, 2, 3, 4, 5],
        srf_in_dim=[0, 1, 2, 3, 4, 5, 8],
        crv_emb_dim=64,
        srf_emb_dim=64,
        
        match_embed_dim=128):
        super().__init__()
        
        self.sbgcn = SBGCN(f_in_width, l_in_width, e_in_width, v_in_width, out_width, k, use_uvnet_features=use_uvnet_features, crv_in_dim=crv_in_dim, srf_in_dim=srf_in_dim, crv_emb_dim=crv_emb_dim, srf_emb_dim=srf_emb_dim)
        self.mlp = Linear(out_width, match_embed_dim)


    def forward(self, brep1, brep2, matches=None):
        #TODO: use known matches somehow

        ids1 = brep1.flat_topos_to_graph_idx.flatten()
        ids2 = brep2.flat_topos_to_graph_idx.flatten()
        _, _, x_f_1, _, x_e_1, x_v_1 = self.sbgcn(brep1)
        _, _, x_f_2, _, x_e_2, x_v_2 = self.sbgcn(brep2)

        unique_ids = ids1.unique()
        #find maximum number of topologies within the batch
        maxcount1 = max([sum(ids1 == idx) for idx in unique_ids])
        maxcount2 = max([sum(ids2 == idx) for idx in unique_ids])

        allfmatches = []
        allematches = []
        allvmatches = []
        for idx in unique_ids:
            x_f_1_pad = simplepad(x_f_1[ids1 == idx], 0, maxcount1)
            x_f_2_pad = simplepad(x_f_2[ids2 == idx], 0, maxcount2)
            x_e_1_pad = simplepad(x_e_1[ids1 == idx], 0, maxcount1)
            x_e_2_pad = simplepad(x_e_2[ids2 == idx], 0, maxcount2)
            x_v_1_pad = simplepad(x_v_1[ids1 == idx], 0, maxcount1)
            x_v_2_pad = simplepad(x_v_2[ids2 == idx], 0, maxcount2)
            fmatches = outermatch(x_f_1_pad, x_f_2_pad)
            ematches = outermatch(x_e_1_pad, x_e_2_pad)
            vmatches = outermatch(x_v_1_pad, x_v_2_pad)
            allfmatches.append(fmatches)
            allematches.append(ematches)
            allvmatches.append(vmatches)

        allfmatches = torch.stack(allfmatches)
        allematches = torch.stack(allematches)
        allvmatches = torch.stack(allvmatches)

        return allfmatches, allematches, allvmatches

