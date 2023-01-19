import pytorch_lightning as pl
from brepmatching.models import PairEmbedder
import torch
from torch.nn import CrossEntropyLoss, LogSoftmax, Parameter, Linear, Sequential, ReLU, Sigmoid, BCEWithLogitsLoss
from torchmetrics import MeanMetric
import numpy as np
import torch.nn.functional as F
from brepmatching.utils import (
    plot_the_fives,
    plot_multiple_metrics,
    plot_tradeoff,
    NUM_METRICS,
    METRIC_COLS,
    TOPO_KINDS,
    add_match_to_frontier,
    greedy_match_2,
    count_batches,
    Running_avg,
    separate_batched_matches,
    compute_metrics_from_matches,
    precompute_adjacency,
    propagate_adjacency,
    unzip_hetdata
)
from brepmatching.loss import *
from automate import HetData
import pandas as pd
from typing import Any, Optional

#from torch.profiler import profile, record_function, ProfilerActivity

class WeightedBCELoss(torch.nn.Module):
    """
    Binary Cross Entropy Loss but weighted for target 0.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        self.loss = BCEWithLogitsLoss(reduction="sum")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss(x[y == 1], y[y == 1]) + self.weight * self.loss(x[y == 0], y[y == 0])

class VotingNetwork(torch.nn.Module):
    def __init__(self, 
                 in_width: int = 128,
                 out_width: int = 2):
        super().__init__()
        self.mlp = Linear(in_width, out_width)
        self.temp = torch.nn.Parameter(torch.tensor(1.))
    
    def forward(self, x: torch.Tensor):
        x = self.mlp(x)
        return F.softmax(x / self.temp, dim=-1)


class MatchingModel(pl.LightningModule):

    def __init__(self,
        # model hyperparams
        f_in_width: int = 62,
        l_in_width: int = 38,
        e_in_width: int = 72,
        v_in_width: int = 3,
        sbgcn_size: int = 64,
        fflayers: int = 6,
        batch_norm: bool = False,
        use_uvnet_features: bool = False,
        srf_emb_dim: int = 64,
        crv_emb_dim: int = 64,

        mp_exact_matches: bool = False,
        mp_overlap_matches: bool = False,
        use_onshape: str = "none",                  # "none", "voting", "aggregate"
        bce_loss_weight: float = 1.0,

        # inference time params
        threshold: float = 0.75,                    # default threshold
        init_strategy: str = "exact",               # "exact" or "overlap"

        # what to execute
        test_greedy: bool = True,
        test_iterative_vs_threshold: bool = True,
        num_thresholds: int = 10,                   # number of thresholds for metrics v. thresh
        test_adjacency: bool = False,
        val_greedy: bool = False,
        val_iter: bool = False,
        ):
        super().__init__()
        
        self.pair_embedder = PairEmbedder(
            s_face=f_in_width,
            s_loop=l_in_width,
            s_edge=e_in_width,
            s_vert=v_in_width,
            embedding_size=sbgcn_size,
            k=fflayers,
            batch_norm=batch_norm,
            mp_exact_matches=mp_exact_matches,
            mp_overlap_matches=mp_overlap_matches,
            mp_cur_matches=True, # TODO: expose parameter
            use_uvnet_features=use_uvnet_features,
            crv_emb_dim=crv_emb_dim,
            srf_emb_dim=srf_emb_dim)

        self.batch_norm = batch_norm

        # TODO: expose variables
        self.mlp = Sequential(
            Linear(sbgcn_size * 2, sbgcn_size),
            ReLU(),
            Linear(sbgcn_size, 1)
        )
        self.loss = WeightedBCELoss(weight=bce_loss_weight) if bce_loss_weight != 1.0 else BCEWithLogitsLoss(reduction="sum")

        self.use_onshape = use_onshape
        assert(use_onshape in ["none", "voting", "aggregate"])
        if use_onshape == "voting":
            self.voting_network = VotingNetwork(sbgcn_size * 2, 2)
        elif use_onshape == "aggregate":
            self.os_projector = Linear(1, sbgcn_size * 2)

        # inference params
        assert(init_strategy in ["exact", "overlap"])
        self.init_strategy = init_strategy
        self.threshold = threshold

        # what to execute
        self.num_thresholds = num_thresholds
        self.thresholds = np.linspace(0.0, 1.0, num_thresholds + 1) if num_thresholds > 0 else np.array([0.5])
        self.test_greedy = test_greedy
        self.test_iterative_vs_threshold = test_iterative_vs_threshold
        self.test_adjacency = test_adjacency
        self.val_iter = val_iter
        self.val_greedy = val_greedy


        self.save_hyperparameters()
        print(self.hparams)
    
    def forward(self,
                data: HetData,
                masks_bool: dict[str, torch.Tensor],
                os_data: Optional[dict[str, torch.Tensor]] = None   # onshape logits -100 or 100
            ) -> dict[str, torch.Tensor]:
        """
        Returns a dict of ("f", "e", "v") to scores (nl, nr) tensor.

        `data` must have "cur_{kinds}_matches".
        `data` will NOT be modified.
        """
        left_emb, right_emb = self.pair_embedder(data)

        left = {}
        right = {}
        left["f"], left["e"], left["v"] = left_emb
        right["f"], right["e"], right["v"] = right_emb

        scores = {}
        for _, _, k in TOPO_KINDS:
            left_k = left[k] # shape (nl, 64)
            right_k = right[k] # shape (nr, 64)

            mask_k = masks_bool[k]

            # create pairwise concatenation
            grid_l, grid_r = torch.meshgrid(torch.arange(left_k.shape[0], device=self.device),
                                            torch.arange(right_k.shape[0], device=self.device),
                                            indexing="ij")
            pw_emb = torch.cat((left_k[grid_l[mask_k]], right_k[grid_r[mask_k]]), dim=-1) # shape (nl, nr, 128)

            sc = torch.zeros(mask_k.shape, device=self.device) # shape (nl, nr)

            if self.use_onshape == "voting":
                assert(os_data is not None)
                votes = self.voting_network(pw_emb).squeeze(-1)
                os_logits_ = os_data[k][mask_k]
                our_logits = self.mlp(pw_emb).squeeze(-1)
                combined_logits = (torch.stack([our_logits, os_logits_], dim=-1) * votes).sum(dim=-1)
            elif self.use_onshape == "aggregate":
                assert(os_data is not None)
                emb = self.os_projector(os_data[k][mask_k].unsqueeze(-1))
                combined_logits = self.mlp(pw_emb + emb).squeeze(-1)
            else:
                combined_logits = self.mlp(pw_emb).squeeze(-1)

            sc[mask_k] = combined_logits
            scores[k] = sc
        return scores

    def init_masks(self, data: HetData) -> dict[str, torch.Tensor]:
        """
        Return (nl, nr) matrix M. M_ij = b iff i and j belongs to batch b otherwise -1.

        `data` will NOT be modified.
        """
        num_batches = data.num_graphs
        masks = {}
        for kinds, _, k in TOPO_KINDS:
            x = torch.full((data[f"left_{kinds}"].shape[0], data[f"right_{kinds}"].shape[0]), -1,
                           dtype=torch.int, device=self.device)
            l_offset = 0
            r_offset = 0
            for b in range(num_batches):
                l_count = (data[f"left_{kinds}_batch"] == b).sum()
                r_count = (data[f"right_{kinds}_batch"] == b).sum()
                x[l_offset:l_offset + l_count, r_offset:r_offset + r_count] = b
                l_offset += l_count
                r_offset += r_count
            masks[k] = x
        return masks
    
    def init_gt_scores(self, data: HetData) -> dict[str, torch.Tensor]:
        """
        `data` will NOT be modified.
        """
        gt_scores = {}
        for kinds, _, k in TOPO_KINDS:
            score = torch.zeros((data[f"left_{kinds}"].shape[0], data[f"right_{kinds}"].shape[0]), device=self.device)
            score[data[f"{kinds}_matches"][0], data[f"{kinds}_matches"][1]] = 1.0
            gt_scores[k] = score
        return gt_scores
    
    def get_os_data(self, data: HetData) -> Optional[dict[str, torch.Tensor]]:
        """
        Returns the logits of onshape match -100 or 100.
        `data` will NOT be modified.
        """
        if self.use_onshape == "none":
            return None
        elif self.use_onshape == "voting" or self.use_onshape == "aggregate":
            os_logits = {}
            for kinds, _, k in TOPO_KINDS:
                logits = torch.full((data[f"left_{kinds}"].shape[0], data[f"right_{kinds}"].shape[0]), -100., device=self.device)
                logits[data[f"os_bl_{kinds}_matches"][0], data[f"os_bl_{kinds}_matches"][1]] = 100.
                os_logits[k] = logits
            return os_logits
        else:
            raise NotImplementedError()


    def init_cur_match(self,
                       data: HetData,
                       mask: dict[str, torch.Tensor],
                       strategy: str) -> dict[str, torch.Tensor]:
        """
        Initialize matches and attach to `data` and modify `mask` on the corresponding matches.

        Return a dict of matches initialized.

        `data` and `mask` will be modified.
        """
        all_matches = {}
        for kinds, _, k in TOPO_KINDS:
            if strategy == "random":
                n_gt_matches = data[f"{kinds}_matches"].shape[1]
                indices = torch.randperm(n_gt_matches)[:n_gt_matches // 2]
                matches = data[f"{kinds}_matches"][:, indices]
            elif strategy == "exact":
                matches = data[f"bl_exact_{kinds}_matches"].clone()
            elif strategy == "overlap":
                matches = data[f"bl_exact_{kinds}_matches"].clone()
                if k != "v":
                    # no overlap for vertices
                    matches = torch.cat([matches, data[f"bl_overlap_{kinds}_matches"].clone()], dim=-1)

            else:
                raise NotImplementedError()
            setattr(data, f"cur_{kinds}_matches", matches)
            data.__edge_sets__[f"cur_{kinds}_matches"] = [f"left_{kinds}", f"right_{kinds}"]
            mask[k][matches[0], :] = -1
            mask[k][:, matches[1]] = -1
            all_matches[k] = matches
        return all_matches

    def compute_loss(self,
                     scores: dict[str, torch.Tensor],
                     gt_scores: dict[str, torch.Tensor],
                     masks_bool: dict[str, torch.Tensor]) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device)
        numel = 0
        for _, _, k in TOPO_KINDS:
            mask_k = masks_bool[k]
            loss += self.loss(scores[k][mask_k], gt_scores[k][mask_k])
            numel += scores[k][mask_k].numel()
        return loss / numel if numel != 0 else torch.tensor(0.0, device=self.device)

    def do_iteration(self, data: HetData, threshold: float, init_strategy: str, use_adjacency: bool) -> tuple[torch.Tensor, HetData]:
        """
        Do the iteration matching the highest scored candidate until the score is below the threshold

        Returns the loss tensor, `HetData` containing "cur_{kinds}_matches" for further evaluation

        Not backwardable.
        `data` will be modified.
        """
        batch_size = count_batches(data)

        cur_masks = self.init_masks(data)

        # setup ground truth scores
        gt_scores = self.init_gt_scores(data)
        os_data = self.get_os_data(data)

        # prepare current match (to coincident)
        init_matches = self.init_cur_match(data, cur_masks, init_strategy)

        if use_adjacency:
            data_l, data_r = unzip_hetdata(data)
            # precomputed adjacency data
            adj_data_l = precompute_adjacency(data_l)
            adj_data_r = precompute_adjacency(data_r)

            # active frontier
            adj_l: dict[str, torch.Tensor] = {}
            adj_r: dict[str, torch.Tensor] = {}
            for kinds, _, k in TOPO_KINDS:
                adj_l[k] = torch.zeros(data_l[kinds].shape[0], dtype=torch.bool, device=self.device)
                adj_r[k] = torch.zeros(data_r[kinds].shape[0], dtype=torch.bool, device=self.device)

            # init active frontier with init match
            for kinds, _, k in TOPO_KINDS:
                for i in range(init_matches[k].shape[1]):
                    add_match_to_frontier(int(init_matches[k][0, i].item()),
                                          int(init_matches[k][1, i].item()),
                                          k,
                                          adj_data_l,
                                          adj_data_r,
                                          adj_l, adj_r)
                

        
        loss = torch.tensor(0.0, device=self.device)
        n_iter = 0
        changed = True
        while changed:
            cur_masks_bool = {k: (cur_masks[k] != -1) for _, _, k in TOPO_KINDS}
            if use_adjacency:
                for _, _, k in TOPO_KINDS:
                    adj_mask = adj_l[k].expand(adj_r[k].shape[0], -1).T.logical_and(
                               adj_r[k].expand(adj_l[k].shape[0], -1))
                    cur_masks_bool[k].logical_and_(adj_mask)

            # score candidates
            scores_logits = self(data, cur_masks_bool, os_data)
            scores = {k: torch.sigmoid(scores_logits[k]) for k in scores_logits}

            # compute loss
            loss += self.compute_loss(scores_logits, gt_scores, cur_masks_bool)
            n_iter += 1

            changed = False
            for b in range(batch_size):
                # find max score
                mx_score = -1.0
                mx_k = "x"
                mx_kinds = "xxx"
                for kinds, _, k in TOPO_KINDS:
                    tmp = scores[k][cur_masks[k] == b]
                    if tmp.numel() != 0:
                        cur_mx = tmp.max().item()
                        if cur_mx > mx_score:
                            mx_score = cur_mx
                            mx_k = k
                            mx_kinds = kinds

                if mx_score <= threshold:
                    continue

                changed = True

                # take first maximum score
                l, r = (int(x.item()) for x in (scores[mx_k] == mx_score).logical_and(cur_masks[mx_k] == b).nonzero()[0])
                
                lb = data[f"left_{mx_kinds}_batch"]
                rb = data[f"right_{mx_kinds}_batch"]
                assert(lb[l] == rb[r])

                # add (l, r) to matches
                setattr(data, f"cur_{mx_kinds}_matches",
                    torch.cat((data[f"cur_{mx_kinds}_matches"], torch.tensor([[l], [r]], device=self.device)), dim=-1))

                if use_adjacency:
                    add_match_to_frontier(l, r, mx_k, adj_data_l, adj_data_r, adj_l, adj_r)

                # do not consider l and r again
                cur_masks[mx_k][l, :] = -1
                cur_masks[mx_k][:, r] = -1
        
        return loss / n_iter, data

    def do_once(self, data: HetData, init_strategy: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        `data` will be modified.
        """
        masks = self.init_masks(data)
        gt_scores = self.init_gt_scores(data)
        os_data = self.get_os_data(data)

        # prepare current match
        self.init_cur_match(data, masks, init_strategy)
        
        masks_bool = {k: (masks[k] != -1) for _, _, k in TOPO_KINDS}
        scores_logits = self(data, masks_bool, os_data)
        scores = {k: torch.sigmoid(scores_logits[k]) for k in scores_logits}
        loss = self.compute_loss(scores_logits, gt_scores, masks_bool)
        return loss, scores

    def do_greedy_and_compute_metric(self, data: HetData, init_strategy: str) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
        """
        `data` will be modified.
        """
        unseen_loss, scores_mtx = self.do_once(data, init_strategy)
        masks = self.init_masks(data)
        init_matches = self.init_cur_match(data, masks, init_strategy)
        all_metrics = {}
        for kinds, _, k in TOPO_KINDS:
            matches, scores = greedy_match_2(scores_mtx[k], masks[k] != -1)
            metrics = []
            for threshold in self.thresholds:
                cur_metrics = compute_metrics_from_matches(data, kinds,
                        torch.cat((init_matches[k], matches[:, scores > threshold]), dim=1))
                metrics.append(cur_metrics)
            all_metrics[k] = np.array(metrics)
        return unseen_loss, all_metrics



    def training_step(self, data, batch_idx):
        loss, _ = self.do_once(data.clone(), "random")

        batch_size=count_batches(data)
        self.log('train_loss/step', loss, on_step=True, on_epoch=False, batch_size=batch_size)
        self.log('train_loss/epoch', loss, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss


    def validation_step(self, data, batch_idx):
        batch_size = count_batches(data)

        # same loss with training
        loss, _ = self.do_once(data.clone(), "random")
        self.log('val_loss', loss, batch_size=batch_size)

        # iterative
        if self.val_iter:
            iter_loss, data_after = self.do_iteration(data.clone(), self.threshold, self.init_strategy, False)
            self.log('val_iter_loss', iter_loss, batch_size=batch_size)
            for kinds, _, _ in TOPO_KINDS:
                self.log_metrics(data_after, kinds, "val_iter")

        output = {}

        # greedy
        if self.val_greedy:
            unseen_loss, output["greedy"] = self.do_greedy_and_compute_metric(data.clone(), self.init_strategy)
            self.log('val_greedy_loss', unseen_loss, batch_size=batch_size)

        return output

    def test_step(self, data, batch_idx):
        batch_size = count_batches(data)

        # same loss with training
        loss, scores = self.do_once(data.clone(), "random")
        self.log('test_loss', loss, batch_size=batch_size)

        # iterative
        if not self.test_iterative_vs_threshold:
            loss, data_after = self.do_iteration(data.clone(), self.threshold, self.init_strategy, False)
            self.log('test_iter_loss', loss, batch_size=batch_size)
            for kinds, _, _ in TOPO_KINDS:
                self.log_metrics(data_after, kinds, "test_iter")
            if self.test_adjacency:
                loss, data_after = self.do_iteration(data.clone(), self.threshold, self.init_strategy, True)
                self.log('test_iter_adj_loss', loss, batch_size=batch_size)
                for kinds, _, _ in TOPO_KINDS:
                    self.log_metrics(data_after, kinds, "test_iter_adj")


        output = {}

        # greedy
        if self.test_greedy:
            unseen_loss, output["greedy"] = self.do_greedy_and_compute_metric(data.clone(), self.init_strategy)
            self.log('test_greedy_loss', unseen_loss, batch_size=batch_size)

        # iterative v threshold
        if self.test_iterative_vs_threshold:
            all_metrics: dict[str, Any] = { "f": [], "e": [], "v": [] }
            mid_thresh = len(self.thresholds) // 2
            for i, threshold in enumerate(self.thresholds):
                iter_loss, data_after = self.do_iteration(data.clone(), threshold, self.init_strategy, False)
                for kinds, _, k in TOPO_KINDS:
                    if i == mid_thresh:
                        self.log_metrics(data_after, kinds, "test_iter")
                    metrics = compute_metrics_from_matches(data_after, kinds, data_after[f"cur_{kinds}_matches"])
                    all_metrics[k].append(metrics)
            for _, _, k in TOPO_KINDS:
                all_metrics[k] = np.array(all_metrics[k])
            output["iter"] = all_metrics
            if self.test_adjacency:
                all_metrics: dict[str, Any] = { "f": [], "e": [], "v": [] }
                for i, threshold in enumerate(self.thresholds):
                    iter_loss, data_after = self.do_iteration(data.clone(), threshold, self.init_strategy, True)
                    for kinds, _, k in TOPO_KINDS:
                        if i == mid_thresh:
                            self.log_metrics(data_after, kinds, "test_iter_adj")
                        metrics = compute_metrics_from_matches(data_after, kinds, data_after[f"cur_{kinds}_matches"])
                        all_metrics[k].append(metrics)
                for _, _, k in TOPO_KINDS:
                    all_metrics[k] = np.array(all_metrics[k])
                output["iter_adj"] = all_metrics

        return output
    

    def validation_epoch_end(self, outputs):
        if self.val_greedy:
            self.log_metrics_v_thresh(outputs, "greedy", "val_greedy")

    def test_epoch_end(self, outputs):
        if self.test_greedy:
            self.log_metrics_v_thresh(outputs, "greedy", "test_greedy", save=True)
        if self.test_iterative_vs_threshold:
            self.log_metrics_v_thresh(outputs, "iter", "test_iter", save=True)
            if self.test_adjacency:
                self.log_metrics_v_thresh(outputs, "iter_adj", "test_iter_adj", save=True)

    
    def log_metrics(self, data: HetData, kinds: str, prefix: str):
        batch_size = count_batches(data)
        true_pos, true_neg, missed, incorrect, false_pos, \
            precision, recall = compute_metrics_from_matches(data, kinds, data[f"cur_{kinds}_matches"])

        self.log(f"{prefix}_correct/{kinds}", true_pos + true_neg, batch_size=batch_size)
        self.log(f"{prefix}_missed/{kinds}", missed, batch_size=batch_size)
        self.log(f"{prefix}_incorrect/{kinds}", incorrect + false_pos, batch_size=batch_size)
        self.log(f"{prefix}_precision/{kinds}", precision, batch_size=batch_size)
        self.log(f"{prefix}_recall/{kinds}", recall, batch_size=batch_size)

    def log_metrics_v_thresh(self, outputs, algo: str, prefix: str, save: bool = False):
        """
        algo: "greedy" or "iter"
        """
        all_metrics = {
            "f": np.zeros_like(outputs[0][algo]["f"]),
            "e": np.zeros_like(outputs[0][algo]["e"]),
            "v": np.zeros_like(outputs[0][algo]["v"]),
        }
        for output in outputs:
            for _, _, k in TOPO_KINDS:
                all_metrics[k] += output[algo][k]
        
        for kinds, _, k in TOPO_KINDS:
            all_metrics[k] /= len(outputs)

            true_pos, true_neg, missed, incorrect, false_pos, \
                precision, recall \
                    = (all_metrics[k][:, i] for i in range(NUM_METRICS))
            
            fig_all = plot_the_fives(true_pos, true_neg, missed, incorrect, false_pos,
                                     self.thresholds, f"Metrics vs. Threshold ({kinds})")
        
            fig_precrecall_flat = plot_multiple_metrics({
                'precision': precision,
                'recall': recall
            }, self.thresholds, f"Precision & Recall vs. Threshold ({kinds})")

            label_indices = [0, len(self.thresholds) // 2, -1]
            fig_precision_recall = plot_tradeoff(
                recall, precision, self.thresholds, label_indices,
                'Recall', 'Precision', f" ({kinds})")

            self.logger.experiment.add_figure(f'{prefix}_metric_v_thresh/{kinds}', fig_all, self.current_epoch)
            self.logger.experiment.add_figure(f'{prefix}_precision_recall_v_thresh/{kinds}', fig_precrecall_flat, self.current_epoch)
            self.logger.experiment.add_figure(f'{prefix}_precision_v_recall/{kinds}', fig_precision_recall, self.current_epoch)
        
        if save:
            entries = []
            for kinds, _, k in TOPO_KINDS:
                for threshold, metrics in zip(self.thresholds, all_metrics[k]):
                    entries.append([threshold, *metrics, kinds])
            df = pd.DataFrame(entries, columns=["Threshold", *METRIC_COLS, "Kind"])
            cur_algo = algo
            if self.init_strategy == "overlap":
                cur_algo += "_ovl"
            df.to_csv(f"{self.logger.log_dir}/{cur_algo}.csv")


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