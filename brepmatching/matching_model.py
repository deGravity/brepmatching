import pytorch_lightning as pl
from brepmatching.models import PairEmbedder
import torch
from torch.nn import CrossEntropyLoss, LogSoftmax
from torchmetrics import MeanMetric

from torch.profiler import profile, record_function, ProfilerActivity



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

    def sample_matches(self, data, topo_type, device='cuda'):
        with torch.no_grad():
            num_batches = getattr(data, 'left_faces_batch')[-1]+1 #Todo: is there a better way to count batches?
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
            keepmask = torch.ones(getattr(data, topo_type + '_matches_batch').shape[0], dtype=torch.bool)
            for batch, (offset, match_offset) in enumerate(zip(batch_offsets, match_batch_offsets)):
                batch_size = (getattr(data, 'right_' + topo_type + '_batch') == batch).sum()
                match_batch_size = (getattr(data, topo_type + '_matches_batch') == batch).sum()
                if batch_size > 1:
                    perms_batch = []
                    for m in range(match_batch_size):
                        match_index = m + match_offset
                        perm = torch.randperm(batch_size, device=device) + offset
                        perm = perm[perm != getattr(data, topo_type + '_matches')[1,match_index]]
                        perms_batch.append(perm)
                    allperms += perms_batch
                else:
                    keepmask[match_offset:match_offset+match_batch_size] = 0
            mincount = min([len(perm) for perm in allperms])
            mincount = min(mincount, self.num_negative)
            allperms = [perm[:mincount] for perm in allperms]
            allperms = torch.stack(allperms)
        return allperms, keepmask #keepmask: mask of which matches to keep (ones belonging to batches with too few right topos are disabled)
    
    def compute_loss(self, allperms, data, f_orig, f_var, topo_type, mask):
        matches_masked = getattr(data, topo_type + '_matches')[:, mask]
        f_orig_matched = f_orig[matches_masked[0]]
        f_var_matched = f_var[matches_masked[1]]
        f_matched_sim = torch.sum(f_orig_matched * f_var_matched, dim=-1)

        f_var_unmatched = f_var[allperms]
        f_orig_unmatched = f_orig_matched.expand(f_var_unmatched.shape[1], f_var_unmatched.shape[0], f_var_unmatched.shape[2]).transpose(0, 1)
        f_unmatched_sim = torch.sum(f_orig_unmatched * f_var_unmatched, dim=-1)

        f_sim = torch.cat([f_matched_sim.unsqueeze(-1), f_unmatched_sim], dim=1)
        logits = self.softmax(f_sim / self.temperature)
        labels = torch.zeros_like(logits)
        labels[:,0] = 1
        return self.loss(logits, labels)
    

    def training_step(self, data, batch_idx):
        face_allperms, faces_match_mask = self.sample_matches(data, 'faces', device=data.left_faces.device)
        edge_allperms, edges_match_mask = self.sample_matches(data, 'edges', device=data.left_faces.device)
        vert_allperms, verts_match_mask = self.sample_matches(data, 'vertices', device=data.left_faces.device)
        (f_orig, e_orig, v_orig), (f_var, e_var, v_var) = self(data)
        f_loss = self.compute_loss(face_allperms, data, f_orig, f_var, 'faces', faces_match_mask)
        e_loss = self.compute_loss(edge_allperms, data, e_orig, e_var, 'edges', edges_match_mask)
        v_loss = self.compute_loss(vert_allperms, data, v_orig, v_var, 'vertices', verts_match_mask)
        loss = f_loss + e_loss + v_loss
        self.log('train_loss/step', loss, on_step=True, on_epoch=False)
        self.log('train_loss/epoch', loss, on_step=False, on_epoch=True)
        return loss


    def validation_step(self, data, batch_idx):
        face_allperms, faces_match_mask = self.sample_matches(data, 'faces', device=data.left_faces.device)
        edge_allperms, edges_match_mask = self.sample_matches(data, 'edges', device=data.left_faces.device)
        vert_allperms, verts_match_mask = self.sample_matches(data, 'vertices', device=data.left_faces.device)
        (f_orig, e_orig, v_orig), (f_var, e_var, v_var) = self(data)
        f_loss = self.compute_loss(face_allperms, data, f_orig, f_var, 'faces', faces_match_mask)
        e_loss = self.compute_loss(edge_allperms, data, e_orig, e_var, 'edges', edges_match_mask)
        v_loss = self.compute_loss(vert_allperms, data, v_orig, v_var, 'vertices', verts_match_mask)
        loss = f_loss + e_loss + v_loss
        self.log('val_loss', loss)

        #with profile(activities=[
        #        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        #    with record_function("model_inference"):
        f_acc = self.log_metrics(data, f_orig, f_var, 'faces')
        e_acc = self.log_metrics(data, e_orig, e_var, 'edges')
        v_acc = self.log_metrics(data, v_orig, v_var, 'vertices')
    
        #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


    def test_step(self, data, batch_idx):
        face_allperms, faces_match_mask = self.sample_matches(data, 'faces', device=data.left_faces.device)
        edge_allperms, edges_match_mask = self.sample_matches(data, 'edges', device=data.left_faces.device)
        vert_allperms, verts_match_mask = self.sample_matches(data, 'vertices', device=data.left_faces.device)
        (f_orig, e_orig, v_orig), (f_var, e_var, v_var) = self(data)
        f_loss = self.compute_loss(face_allperms, data, f_orig, f_var, 'faces', faces_match_mask)
        e_loss = self.compute_loss(edge_allperms, data, e_orig, e_var, 'edges', edges_match_mask)
        v_loss = self.compute_loss(vert_allperms, data, v_orig, v_var, 'vertices', verts_match_mask)
        loss = f_loss + e_loss + v_loss
        self.log('test_loss', loss)


    
    def log_metrics(self, data, orig_emb, var_emb, topo_type):
        orig_emb_match = orig_emb[getattr(data, topo_type + '_matches')[0]]

        num_batches = getattr(data, 'left_faces_batch')[-1]+1
        batch_right_inds = []
        batch_right_feats = []
        for j in range(num_batches):
            inds = (getattr(data, 'right_'+topo_type+'_batch') == j).nonzero().flatten()
            batch_right_inds.append(inds)
            batch_right_feats.append(var_emb[inds].T)

        matches = []
        for m in range(orig_emb_match.shape[0]):
            curr_batch = getattr(data, topo_type + '_matches_batch')[m]

            dists =  orig_emb_match[m] @ batch_right_feats[curr_batch]
            maxind = torch.argmax(dists)
            matches.append(batch_right_inds[curr_batch][maxind])

        matches = torch.stack(matches)
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
