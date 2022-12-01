import pytorch_lightning as pl
from .matching import Matcher
import torch

class MatchingModel(pl.LightningModule):

    def __init__(self,
        f_in_width: int = 60,
        l_in_width: int = 38,
        e_in_width: int = 68,
        v_in_width: int = 3,
        sbgcn_size: int = 64,
        fflayers: int = 1,
        use_uvnet_features: bool = False,
        match_embed_dim: int = 128):
        super().__init__()

        self.match = Matcher(
            f_in_width,
            l_in_width,
            e_in_width,
            v_in_width,
            sbgcn_size,
            fflayers,
            use_uvnet_features=use_uvnet_features,
            match_embed_dim=match_embed_dim
        )



        self.save_hyperparameters()
    

    def forward(self, brep1, brep2, matches=None):
        return self.match(brep1, brep2, matches)

    
    def training_step(self, batch, batch_idx):
        brep1, brep2, matches = batch
        fm, em, vm = self(batch)
        loss = self.compute_loss(fm, em, vm)
        return loss

    def validation_step(self, batch, batch_idx):
        pass
    

    def test_step(self, batch, batch_idx):
        pass


    def compute_loss(self, fm, em, vm, labels):
        return 0

        
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
