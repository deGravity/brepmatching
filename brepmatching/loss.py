import torch
from torch.nn import CrossEntropyLoss, Module, Parameter
from brepmatching.utils import logsumexp

#these loss functions assume the features are in the form [sim_pos, sim_neg_1, sim_neg_2, ...] where each `sim` comprises a column of the matrix
#i.e. the target is [1, 0, 0, ...] for each row

class N_pairs_loss(Module):
    def __init__(self, temperature=-1):
        super().__init__()
        self.loss = CrossEntropyLoss()
        if temperature < 0:
            self.temperature = Parameter(torch.tensor(0.07))
        else:
            self.temperature = temperature

    def __call__(self, sims):
        logits = sims  * torch.exp(self.temperature)
        labels = torch.zeros_like(logits[:,0], dtype=torch.int64)
        return self.loss(logits, labels)


class TupletMarginLoss(Module):
    def __init__(self, margin=0.1, scale=64):
        super().__init__()
        self.margin = margin
        self.scale = scale
    
    def __call__(self, sims):
        sims[:,0] = torch.cos(torch.acos(sims[:,0] - self.margin))

        inside_exp = self.scale * (sims[:,1:] - sims[:,0:1])
        return logsumexp(inside_exp, add_one=True, dim=1).mean()