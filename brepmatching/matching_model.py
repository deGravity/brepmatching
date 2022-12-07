import pytorch_lightning as pl
from brepmatching.models import PairEmbedder
import torch
from torch.nn import CrossEntropyLoss, LogSoftmax
from torchmetrics import MeanMetric


class MatchingModel(pl.LightningModule):

    def __init__(self,
        f_in_width: int = 62,
        l_in_width: int = 38,
        e_in_width: int = 72,
        v_in_width: int = 3,
        sbgcn_size: int = 64,
        fflayers: int = 6,

        #use_uvnet_features: bool = False,
        temperature: float = 100.0, #temperature normalization factor for contrastive softmax
        num_negative: int = 5
        
        ):
        super().__init__()

        self.pair_embedder = PairEmbedder(f_in_width, l_in_width, e_in_width, v_in_width, sbgcn_size, fflayers)
        self.temperature = temperature
        self.num_negative = num_negative

        self.loss = CrossEntropyLoss()
        self.softmax = LogSoftmax(dim=1)

        # self.faces_accuracy = MeanMetric()
        # self.edges_accuracy = MeanMetric()
        # self.vertices_accuracy = MeanMetric()
        # self.accuracy = MeanMetric()

        self.save_hyperparameters()
    

    def forward(self, data):
        return self.pair_embedder(data)

    def sample_matches(self, data, topo_type):
        with torch.no_grad():
            num_batches = len(getattr(data, 'left_' + topo_type + '_batch').unique()) #Todo: is there a better way to count batches?
            batch_offsets = []
            batch_offset = torch.tensor(0)
            for batch in range(num_batches):
                batch_offsets.append(batch_offset)
                batch_size = (getattr(data, 'right_' + topo_type + '_batch') == batch).sum()
                batch_offset = batch_offset.clone() + batch_size
            
            match_batch_offsets = []
            batch_offset = torch.tensor(0)
            for batch in range(num_batches):
                match_batch_offsets.append(batch_offset)
                batch_size = (getattr(data, topo_type + '_matches_batch') == batch).sum()
                batch_offset = batch_offset.clone() + batch_size

            allperms = []
            for batch, (offset, match_offset) in enumerate(zip(batch_offsets, match_batch_offsets)):
                perms_batch = []
                for m in range((getattr(data, topo_type + '_matches_batch') == batch).sum()):
                    match_index = m + match_offset
                    perm = torch.randperm((getattr(data, 'right_' + topo_type + '_batch') == batch).sum()) + offset
                    perm = perm[perm != getattr(data, topo_type + '_matches')[1,match_index]]
                    perms_batch.append(perm)
                allperms += perms_batch
            mincount = min([len(perm) for perm in allperms])
            mincount = min(mincount, self.num_negative)
            allperms = [perm[:mincount] for perm in allperms]
            allperms = torch.stack(allperms)
        return allperms
    
    def compute_loss(self, face_allperms, data, f_orig, f_var, topo_type):
        f_orig_matched = f_orig[getattr(data, topo_type + '_matches')[0]]
        f_var_matched = f_var[getattr(data, topo_type + '_matches')[1]]
        f_matched_sim = torch.sum(f_orig_matched * f_var_matched, dim=-1)

        f_var_unmatched = f_var[face_allperms]
        f_orig_unmatched = f_orig_matched.expand(f_var_unmatched.shape[1], f_var_unmatched.shape[0], f_var_unmatched.shape[2]).transpose(0, 1)
        f_unmatched_sim = torch.sum(f_orig_unmatched * f_var_unmatched, dim=-1)

        f_sim = torch.cat([f_matched_sim.unsqueeze(-1), f_unmatched_sim], dim=1)
        logits = self.softmax(f_sim / self.temperature)
        labels = torch.zeros_like(logits)
        labels[:,0] = 1
        return self.loss(logits, labels)
    

    def training_step(self, data, batch_idx):
        face_allperms = self.sample_matches(data, 'faces')
        edge_allperms = self.sample_matches(data, 'edges')
        vert_allperms = self.sample_matches(data, 'vertices')
        (f_orig, e_orig, v_orig), (f_var, e_var, v_var) = self(data)
        f_loss = self.compute_loss(face_allperms, data, f_orig, f_var, 'faces')
        e_loss = self.compute_loss(edge_allperms, data, e_orig, e_var, 'edges')
        v_loss = self.compute_loss(vert_allperms, data, v_orig, v_var, 'vertices')
        loss = f_loss + e_loss + v_loss
        self.log('train_loss/step', loss, on_step=True, on_epoch=False)
        self.log('train_loss/epoch', loss, on_step=False, on_epoch=True)
        return loss


    def validation_step(self, data, batch_idx):
        face_allperms = self.sample_matches(data, 'faces')
        edge_allperms = self.sample_matches(data, 'edges')
        vert_allperms = self.sample_matches(data, 'vertices')
        (f_orig, e_orig, v_orig), (f_var, e_var, v_var) = self(data)
        f_loss = self.compute_loss(face_allperms, data, f_orig, f_var, 'faces')
        e_loss = self.compute_loss(edge_allperms, data, e_orig, e_var, 'edges')
        v_loss = self.compute_loss(vert_allperms, data, v_orig, v_var, 'vertices')
        loss = f_loss + e_loss + v_loss
        self.log('val_loss', loss)

        f_acc = self.log_metrics(data, f_orig, f_var, 'faces')
        e_acc = self.log_metrics(data, e_orig, e_var, 'edges')
        v_acc = self.log_metrics(data, v_orig, v_var, 'vertices')


    def test_step(self, data, batch_idx):
        face_allperms = self.sample_matches(data, 'faces')
        edge_allperms = self.sample_matches(data, 'edges')
        vert_allperms = self.sample_matches(data, 'vertices')
        (f_orig, e_orig, v_orig), (f_var, e_var, v_var) = self(data)
        f_loss = self.compute_loss(face_allperms, data, f_orig, f_var, 'faces')
        e_loss = self.compute_loss(edge_allperms, data, e_orig, e_var, 'edges')
        v_loss = self.compute_loss(vert_allperms, data, v_orig, v_var, 'vertices')
        loss = f_loss + e_loss + v_loss
        self.log('test_loss', loss)


    
    def log_metrics(self, data, orig_emb, var_emb, topo_type):
        orig_match = orig_emb[getattr(data, topo_type + '_matches')[0]]
        matches = []
        for m in range(orig_match.shape[0]):
            maxdist = -torch.inf
            maxind = -1
            batch_right_inds = (getattr(data, topo_type + '_matches_batch')[m] == getattr(data, 'right_'+topo_type+'_batch')).nonzero().flatten()
            for j in batch_right_inds:
                dist = torch.dot(orig_match[m], var_emb[j])
                if dist > maxdist:
                    maxdist = dist
                    maxind = j
            matches.append(maxind)
        matches = torch.tensor(matches)
        acc = (matches == getattr(data, topo_type + '_matches')[1]).sum() / len(matches)
        self.log('accuracy/' + topo_type, acc)
        return acc


    def get_callbacks(self):
        callbacks = [
            pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=4, filename="{epoch}-{val_loss:.6f}", mode="min"),
            pl.callbacks.ModelCheckpoint(save_top_k=-1, every_n_epochs=5),
            pl.callbacks.ModelCheckpoint(save_last=True),
        ]
        return callbacks
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return pl.utilities.argparse.from_argparse_args(cls, args, **kwargs)

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = pl.utilities.argparse.add_argparse_args(cls, parent_parser)
        return parser
