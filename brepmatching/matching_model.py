import pytorch_lightning as pl
from brepmatching.models import PairEmbedder
import torch
from torch.nn import CrossEntropyLoss, LogSoftmax


class MatchingModel(pl.LightningModule):

    def __init__(self,
        f_in_width: int = 62,
        l_in_width: int = 38,
        e_in_width: int = 72,
        v_in_width: int = 3,
        sbgcn_size: int = 64,
        fflayers: int = 6,

        #use_uvnet_features: bool = False,
        temperature: float = 1.0, #temperature normalization factor for contrastive softmax
        num_negative: int = 5
        
        ):
        super().__init__()

        self.pair_embedder = PairEmbedder(f_in_width, l_in_width, e_in_width, v_in_width, sbgcn_size, fflayers)
        self.temperature = temperature
        self.num_negative = num_negative

        self.loss = CrossEntropyLoss()
        self.softmax = LogSoftmax(dim=1)

        self.save_hyperparameters()
    

    def forward(self, data):
        return self.pair_embedder(data)

    def sample_matches(self, data):
        with torch.no_grad():
            num_batches = len(data.left_faces_batch.unique()) #Todo: is there a better way to count batches?
            face_batch_offsets = []
            batch_offset = torch.tensor(0)
            for batch in range(num_batches):
                face_batch_offsets.append(batch_offset)
                batch_size = (data.right_faces_batch == batch).sum()
                batch_offset = batch_offset.clone() + batch_size
            
            face_match_batch_offsets = []
            batch_offset = torch.tensor(0)
            for batch in range(num_batches):
                face_match_batch_offsets.append(batch_offset)
                batch_size = (data.face_matches_batch == batch).sum()
                batch_offset = batch_offset.clone() + batch_size

            
            face_allperms = []
            for batch, (face_offset, face_match_offset) in enumerate(zip(face_batch_offsets, face_match_batch_offsets)):
                perms_batch = []
                for m in range((data.face_matches_batch == batch).sum()):
                    match_index = m + face_match_offset
                    perm = torch.randperm((data.right_faces_batch == batch).sum()) + face_offset
                    perm = perm[perm != data.face_matches[1,match_index]]
                    perms_batch.append(perm)
                face_allperms += perms_batch
            mincount = min([len(perm) for perm in face_allperms])
            mincount = min(mincount, self.num_negative)
            face_allperms = [perm[:mincount] for perm in face_allperms]
            face_allperms = torch.stack(face_allperms)
        return face_allperms
    
    def compute_loss(self, face_allperms, data, f_orig, f_var):
        f_orig_matched = f_orig[data.face_matches[0]]
        f_var_matched = f_var[data.face_matches[1]]
        f_matched_sim = torch.sum(f_orig_matched * f_var_matched, dim=-1)

        f_var_unmatched = f_var[face_allperms]
        f_orig_unmatched = f_orig_matched.expand(f_var_unmatched.shape[1], f_var_unmatched.shape[0], f_var_unmatched.shape[2]).transpose(0, 1)
        f_unmatched_sim = torch.sum(f_orig_unmatched * f_var_unmatched, dim=-1)

        f_sim = torch.cat([f_matched_sim.unsqueeze(-1), f_unmatched_sim], dim=1)
        logits = self.softmax(f_sim)
        labels = torch.zeros_like(logits)
        labels[:,0] = 1
        return self.loss(logits, labels)
    

    def training_step(self, data, batch_idx):
        face_allperms = self.sample_matches(data)
        (f_orig, e_orig, v_orig), (f_var, e_var, v_var) = self(data)
        loss = self.compute_loss(face_allperms, data, f_orig, f_var)
        self.log('train_loss/step', loss, on_step=True, on_epoch=False)
        self.log('train_loss/epoch', loss, on_step=False, on_epoch=True)
        return loss


    def validation_step(self, data, batch_idx):
        face_allperms = self.sample_matches(data)
        (f_orig, e_orig, v_orig), (f_var, e_var, v_var) = self(data)
        loss = self.compute_loss(face_allperms, data, f_orig, f_var)
        self.log('val_loss', loss)


    def test_step(self, data, batch_idx):
        face_allperms = self.sample_matches(data)
        (f_orig, e_orig, v_orig), (f_var, e_var, v_var) = self(data)
        loss = self.compute_loss(face_allperms, data, f_orig, f_var)
        self.log('test_loss', loss)


    

        
    def log_metrics(self, batch, preds, mode):
        pass


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
