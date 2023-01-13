import pytorch_lightning as pl
from brepmatching.models import PairEmbedder
import torch
from torch.nn import CrossEntropyLoss, LogSoftmax, Parameter, Linear, Sequential, ReLU, Sigmoid, BCELoss
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
from typing import Any

#from torch.profiler import profile, record_function, ProfilerActivity

class WeightedBCELoss:
    """
    Binary Cross Entropy Loss but weighted for target 0.
    """
    def __init__(self, weight: float = 1.0):
        self.weight = weight
        self.loss = BCELoss(reduction="sum")

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss(x[y == 1], y[y == 1]) + self.weight * self.loss(x[y == 0], y[y == 0])

class MatchingModel(pl.LightningModule):

    def __init__(self,
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

        num_negative: int = 5,
        num_thresholds: int = 10,
        min_topos: int = 1,

        loss: str = 'NPairs', #NPairs, TupletMargin
        temperature: float = -1,
        margin: float = 0.1,
        loss_scale: float = 64,

        log_baselines: bool = False,

        threshold: float = 0.75,
        use_adjacency: bool = False,
        test_greedy: bool = True,
        test_iterative_vs_threshold: bool = True,
        bce_loss_weight: float = 1.0
        ):
        super().__init__()

        self.num_negative = num_negative
        self.num_thresholds = num_thresholds
        self.thresholds = np.linspace(0.0, 1.0, num_thresholds + 1) if num_thresholds > 0 else np.array([0.5])
        self.threshold = threshold
        self.min_topos = min_topos
        self.log_baselines = log_baselines
        
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

        # TODO: expose variables
        self.mlp = Sequential(
            Linear(sbgcn_size * 2, sbgcn_size),
            ReLU(),
            Linear(sbgcn_size, 1),
            Sigmoid()
        )
        self.loss = WeightedBCELoss(weight=bce_loss_weight) if bce_loss_weight != 1.0 else BCELoss(reduction="sum")
        self.test_greedy = test_greedy
        self.test_iterative_vs_threshold = test_iterative_vs_threshold
        self.use_adjacency = use_adjacency

        self.softmax = LogSoftmax(dim=1)
        self.batch_norm = batch_norm

        self.averages = {}
        for topo_type in ['faces', 'edges','vertices']:
            self.averages[topo_type + '_recall'] = Running_avg(num_thresholds)
            self.averages[topo_type + '_precision'] = Running_avg(num_thresholds)
            self.averages[topo_type + '_falsepositives'] = Running_avg(num_thresholds)
            self.averages[topo_type + '_true_positives_and_negatives'] = Running_avg(num_thresholds)
            self.averages[topo_type + '_incorrect_and_falsepositive'] = Running_avg(num_thresholds)
            self.averages[topo_type + '_missed'] = Running_avg(num_thresholds)


        self.save_hyperparameters()
    
    def forward(self, data: HetData, masks_bool: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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
            sc[mask_k] = self.mlp(pw_emb).squeeze(-1)
            scores[k] = sc
        return scores

    def init_masks(self, data: HetData) -> dict[str, torch.Tensor]:
        """
        Return (nl, nr) matrix M. M_ij = b iff i and j belongs to batch b otherwise -1.

        `data` will NOT be modified.
        """
        num_batches = max(data.left_faces_batch[-1], data.right_faces_batch[-1]) + 1
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

    def do_iteration(self, data: HetData, threshold: float) -> tuple[torch.Tensor, HetData]:
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

        # prepare current match (to coincident)
        init_matches = self.init_cur_match(data, cur_masks, "exact")

        if self.use_adjacency:
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
            if self.use_adjacency:
                for _, _, k in TOPO_KINDS:
                    adj_mask = adj_l[k].expand(adj_r[k].shape[0], -1).T.logical_and(
                               adj_r[k].expand(adj_l[k].shape[0], -1))
                    cur_masks_bool[k].logical_and_(adj_mask)

            # score candidates
            scores = self(data, cur_masks_bool)

            # compute loss
            loss += self.compute_loss(scores, gt_scores, cur_masks_bool)
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

                if self.use_adjacency:
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

        # prepare current match
        self.init_cur_match(data, masks, init_strategy)
        
        masks_bool = {k: (masks[k] != -1) for _, _, k in TOPO_KINDS}
        scores = self(data, masks_bool)
        loss = self.compute_loss(scores, gt_scores, masks_bool)
        return loss, scores

    def do_greedy_and_compute_metric(self, data: HetData) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
        """
        `data` will be modified.
        """
        unseen_loss, scores_mtx = self.do_once(data, "exact")
        masks = self.init_masks(data)
        init_matches = self.init_cur_match(data, masks, "exact")
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
        iter_loss, data_after = self.do_iteration(data.clone(), self.threshold)
        self.log('val_iter_loss', iter_loss, batch_size=batch_size)
        for kinds, _, _ in TOPO_KINDS:
            self.log_metrics(data_after, kinds, "val_iter")

        output = {}

        # greedy
        if self.test_greedy:
            unseen_loss, output["greedy"] = self.do_greedy_and_compute_metric(data.clone())
            self.log('val_greedy_loss', unseen_loss, batch_size=batch_size)

        return output

    def test_step(self, data, batch_idx):
        batch_size = count_batches(data)

        # same loss with training
        loss, scores = self.do_once(data.clone(), "random")
        self.log('test_loss', loss, batch_size=batch_size)

        # iterative
        loss, data_after = self.do_iteration(data.clone(), self.threshold)
        self.log('test_iter_loss', loss, batch_size=batch_size)
        for kinds, _, _ in TOPO_KINDS:
            self.log_metrics(data_after, kinds, "test_iter")

        output = {}

        # greedy
        if self.test_greedy:
            unseen_loss, output["greedy"] = self.do_greedy_and_compute_metric(data.clone())
            self.log('test_greedy_loss', unseen_loss, batch_size=batch_size)

        # iterative v threshold
        if self.test_iterative_vs_threshold:
            all_metrics: dict[str, Any] = { "f": [], "e": [], "v": [] }
            for threshold in self.thresholds:
                iter_loss, data_after = self.do_iteration(data.clone(), threshold)
                for kinds, _, k in TOPO_KINDS:
                    metrics = compute_metrics_from_matches(data_after, kinds, data_after[f"cur_{kinds}_matches"])
                    all_metrics[k].append(metrics)
            for _, _, k in TOPO_KINDS:
                all_metrics[k] = np.array(all_metrics[k])
            output["iter"] = all_metrics

        return output
    

    def validation_epoch_end(self, outputs):
        if self.test_greedy:
            self.log_metrics_v_thresh(outputs, "greedy", "val_greedy")

    def test_epoch_end(self, outputs):
        if self.test_greedy:
            self.log_metrics_v_thresh(outputs, "greedy", "test_greedy", save=True)
        if self.test_iterative_vs_threshold:
            self.log_metrics_v_thresh(outputs, "iter", "test_iter", save=True)
    
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
            df.to_csv(f"{self.logger.log_dir}/{prefix}_metrics.csv")


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