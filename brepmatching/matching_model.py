import pytorch_lightning as pl
from brepmatching.models import PairEmbedder
import torch
from torch.nn import CrossEntropyLoss, LogSoftmax, Parameter, Linear, Sequential, ReLU, Sigmoid, BCELoss
from torchmetrics import MeanMetric
import numpy as np
import torch.nn.functional as F
from brepmatching.utils import plot_metric, plot_multiple_metrics, plot_tradeoff, greedy_matching, count_batches, compute_metrics, Running_avg, separate_batched_matches, compute_metrics_2
from brepmatching.loss import *
from automate import HetData

#from torch.profiler import profile, record_function, ProfilerActivity

TOPO_KINDS: list[tuple[str, str, str]] = [
  ("faces", "face", "f"),
  # ("loops", "loop", "l"),
  ("edges", "edge", "e"),
  ("vertices", "vertex", "v")
]

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

        log_baselines: bool = False
        
        ):
        super().__init__()

        self.num_negative = num_negative
        self.num_thresholds = num_thresholds
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
        self.threshold = 0.75
        self.loss = BCELoss(reduction="sum") 

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
    
    def forward(self, data: HetData, masks: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Returns a dict of ("f", "e", "v") to scores (nl, nr) tensor.

        `data` must have "cur_{kinds}_matches"
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

            # create pairwise concatenation
            grid_l, grid_r = torch.meshgrid(torch.arange(left_k.shape[0], device=self.device),
                                            torch.arange(right_k.shape[0], device=self.device))
            pw_emb = torch.cat((left_k[grid_l[masks[k]]], right_k[grid_r[masks[k]]]), dim=-1) # shape (nl, nr, 128)

            sc = torch.zeros(masks[k].shape, device=self.device) # shape (nl, nr)
            sc[masks[k]] = self.mlp(pw_emb).squeeze(-1)
            scores[k] = sc
        return scores

    def init_masks(self, data: HetData) -> dict[str, torch.Tensor]:
        num_batches = max(data.left_faces_batch[-1], data.right_faces_batch[-1]) + 1
        masks = {}
        for kinds, _, k in TOPO_KINDS:
            x = torch.zeros((data[f"left_{kinds}"].shape[0], data[f"right_{kinds}"].shape[0]),
                dtype=torch.bool, device=self.device)
            l_offset = 0
            r_offset = 0
            for b in range(num_batches):
                l_count = (data[f"left_{kinds}_batch"] == b).sum()
                r_count = (data[f"right_{kinds}_batch"] == b).sum()
                x[l_offset:l_offset + l_count, r_offset:r_offset + r_count] = True
                l_offset += l_count
                r_offset += r_count
            masks[k] = x
        return masks
    
    def init_gt_scores(self, data: HetData) -> dict[str, torch.Tensor]:
        gt_scores = {}
        for kinds, _, k in TOPO_KINDS:
            score = torch.zeros((data[f"left_{kinds}"].shape[0], data[f"right_{kinds}"].shape[0]), device=self.device)
            score[data[f"{kinds}_matches"][0], data[f"{kinds}_matches"][1]] = 1.0
            gt_scores[k] = score
        return gt_scores

    def compute_loss(self,
                     scores: dict[str, torch.Tensor],
                     gt_scores: dict[str, torch.Tensor],
                     masks: dict[str, torch.Tensor]) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device)
        numel = 0
        for _, _, k in TOPO_KINDS:
            loss += self.loss(scores[k][masks[k]], gt_scores[k][masks[k]])
            numel += scores[k][masks[k]].numel()
        return loss / numel if numel != 0 else torch.tensor(0.0, device=self.device)


    def do_iteration(self, data: HetData) -> tuple[torch.Tensor, HetData]:
        """
        Do the iteration matching the highest scored candidate until the score is below the threshold

        Not backwardable.

        Returns the loss tensor, `HetData` containing "cur_{kinds}_matches" for further evaluation
        """

        cur_masks = self.init_masks(data)

        # setup ground truth scores
        gt_scores = self.init_gt_scores(data)

        # prepare current match (to coincident)
        for kinds, _, _ in TOPO_KINDS:
            setattr(data, f"cur_{kinds}_matches",
                data[f"bl_exact_{kinds}_matches"].clone())
            data.__edge_sets__[f"cur_{kinds}_matches"] = [f"left_{kinds}", f"right_{kinds}"]
        
        loss = torch.tensor(0.0, device=self.device)
        n_iter = 0
        while True:
            # score candidates
            scores = self(data, cur_masks)

            # find max score
            mx_score = -1.0
            mx_k = "x"
            mx_kinds = "xxx"
            for kinds, _, k in TOPO_KINDS:
                tmp = scores[k][cur_masks[k]]
                if tmp.numel() != 0:
                    cur_mx = tmp.max().item()
                    if cur_mx > mx_score:
                        mx_score = cur_mx
                        mx_k = k
                        mx_kinds = kinds

            # compute loss
            loss += self.compute_loss(scores, gt_scores, cur_masks)
            n_iter += 1

            if mx_score < self.threshold:
                break

            # take first maximum score
            l, r = (int(x.item()) for x in (scores[mx_k] == mx_score).logical_and(cur_masks[mx_k]).nonzero()[0])
            
            lb = data[f"left_{mx_kinds}_batch"]
            rb = data[f"right_{mx_kinds}_batch"]
            assert(lb[l] == rb[r])

            # add (l, r) to matches
            setattr(data, f"cur_{mx_kinds}_matches",
                torch.cat((data[f"cur_{mx_kinds}_matches"], torch.tensor([[l], [r]], device=self.device)), dim=-1))

            # do not consider l and r again
            cur_masks[mx_k][l, :] = False
            cur_masks[mx_k][:, r] = False
        
        return loss / n_iter, data

    def training_step(self, data, batch_idx):
        cur_data = data.clone()
        masks = self.init_masks(cur_data)
        gt_scores = self.init_gt_scores(cur_data)

        # prepare current match (to randomly 50% of gt_match)
        for kinds, _, _ in TOPO_KINDS:
            n_gt_matches = cur_data[f"{kinds}_matches"].shape[1]
            indices = torch.randperm(n_gt_matches)[:n_gt_matches // 2]
            setattr(cur_data, f"cur_{kinds}_matches",
                cur_data[f"{kinds}_matches"][:, indices])
            cur_data.__edge_sets__[f"cur_{kinds}_matches"] = [f"left_{kinds}", f"right_{kinds}"]
        
        scores = self(cur_data, masks)
        loss = self.compute_loss(scores, gt_scores, masks)
        
        assert(not loss.isnan())

        batch_size=count_batches(data)
        self.log('train_loss/step', loss, on_step=True, on_epoch=False, batch_size=batch_size)
        self.log('train_loss/epoch', loss, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss


    def validation_step(self, data, batch_idx):
        loss, data_after = self.do_iteration(data.clone())
        self.log('val_loss', loss, batch_size=count_batches(data))

        for kinds, _, _ in TOPO_KINDS:
            self.log_metrics(data_after, kinds)


    def test_step(self, data, batch_idx):
        loss, data_after = self.do_iteration(data)
        self.log('test_loss', loss, batch_size=count_batches(data))

        for kinds, _, _ in TOPO_KINDS:
            self.log_metrics(data_after, kinds)
    

    def validation_epoch_end(self, outputs):
        # self.log_final_metrics('faces')
        # self.log_final_metrics('edges')
        # self.log_final_metrics('vertices')
        pass


    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)
    

    def _log_baselines(self, data, topo_type):
        batch_size = count_batches(data)
        baseline_matches = getattr(data, 'bl_exact_' + topo_type + '_matches')
        separate_matches = separate_batched_matches(baseline_matches, getattr(data, 'left_'+topo_type+'_batch'), getattr(data, 'right_'+topo_type+'_batch'))
        separate_matches = [match_tensor.T.cpu().numpy() for match_tensor in separate_matches]

        truenegatives, falsepositives, missed,  incorrect, true_positives_and_negatives, incorrect_and_falsepositive, precision, recall, right2left_matched_accuracy = compute_metrics(data, separate_matches, None, topo_type, [-1])
        self.log(topo_type + '/baseline_recall', recall[0], batch_size = batch_size)
        self.log(topo_type + '/baseline_precision', precision[0], batch_size = batch_size)
        self.log(topo_type + '/baseline_falsepositives', falsepositives[0], batch_size = batch_size)
        self.log(topo_type + '/baseline_true_positives_and_negatives', true_positives_and_negatives[0], batch_size = batch_size)
        self.log(topo_type + '/baseline_incorrect_and_falsepositive', incorrect_and_falsepositive[0], batch_size = batch_size)
        self.log(topo_type + '/baseline_missed', missed[0], batch_size = batch_size)


    
    def log_metrics(self, data: HetData, kinds: str):
        # if self.log_baselines:
            # self._log_baselines(data, topo_type)
        
        batch_size = count_batches(data)
        true_neg, false_pos, missed, incorrect, true_pos_and_neg, incorrect_and_false_pos, \
            precision, recall = compute_metrics_2(data, kinds)

        self.log(f"val/{kinds}/true_neg", true_neg, batch_size=batch_size)
        self.log(f"val/{kinds}/false_pos", false_pos, batch_size=batch_size)
        self.log(f"val/{kinds}/missed", missed, batch_size=batch_size)
        self.log(f"val/{kinds}/incorrect", incorrect, batch_size=batch_size)
        self.log(f"val/{kinds}/true_pos_and_neg", true_pos_and_neg, batch_size=batch_size)
        self.log(f"val/{kinds}/incorrect_and_false_pos", incorrect_and_false_pos, batch_size=batch_size)
        self.log(f"val/{kinds}/precision", precision, batch_size=batch_size)
        self.log(f"val/{kinds}/recall", recall, batch_size=batch_size)

        # batch_size = count_batches(data)
        # cur_matches = data[f"cur_{topo_type}_matches"]
        # separate_matches = separate_batched_matches(cur_matches, getattr(data, 'left_'+topo_type+'_batch'), getattr(data, 'right_'+topo_type+'_batch'))
        # separate_matches = [match_tensor.T.cpu().numpy() for match_tensor in separate_matches]

        # truenegatives, falsepositives, missed,  incorrect, true_positives_and_negatives, incorrect_and_falsepositive, precision, recall, right2left_matched_accuracy = compute_metrics(data, separate_matches, None, topo_type, [-1])
        # self.averages[topo_type + '_recall'](recall, batch_size)
        # self.averages[topo_type + '_precision'](precision, batch_size)
        # self.averages[topo_type + '_falsepositives'](falsepositives, batch_size)
        # self.averages[topo_type + '_true_positives_and_negatives'](true_positives_and_negatives, batch_size)
        # self.averages[topo_type + '_incorrect_and_falsepositive'](incorrect_and_falsepositive, batch_size)
        # self.averages[topo_type + '_missed'](missed, batch_size)
        
        # self.log('right2left_matched_accuracy/' + topo_type, right2left_matched_accuracy, batch_size = batch_size)
        # fig_truenegative = plot_metric(truenegatives, thresholds, 'Percent Correct (unmatched)')
        # fig_falsepositive = plot_metric(falsepositives, thresholds, 'Percent False Positive (unmatched)')
        # fig_missed = plot_metric(missed, thresholds, 'Percent Missed (matched)')
        # fig_incorrect = plot_metric(incorrect, thresholds, 'Percent Incorrect (matched)')
        # fig_correct = plot_metric(true_positives_and_negatives, thresholds, 'Percent Correct (all)')
        # fig_incorrect_and_false_positive = plot_metric(incorrect_and_falsepositive, thresholds, 'Percent False Positive or Incorrect (all)')

        # fig_recall = plot_metric(recall, thresholds, 'Recall')
        # fig_precision = plot_metric(precision, thresholds, 'Precision')

        

        # self.logger.experiment.add_figure('truenegative/' + topo_type, fig_truenegative, self.current_epoch)
        # self.logger.experiment.add_figure('falsepositive/' + topo_type, fig_falsepositive, self.current_epoch)
        # self.logger.experiment.add_figure('missed/' + topo_type, fig_missed, self.current_epoch)
        # self.logger.experiment.add_figure('incorrect/' + topo_type, fig_incorrect, self.current_epoch)
        # self.logger.experiment.add_figure('correct/' + topo_type, fig_correct, self.current_epoch)
        # self.logger.experiment.add_figure('incorrect_and_false_positive/' + topo_type, fig_incorrect_and_false_positive, self.current_epoch)
        # self.logger.experiment.add_figure('recall/' + topo_type, fig_recall, self.current_epoch)
        # self.logger.experiment.add_figure('precision/' + topo_type, fig_precision, self.current_epoch)\
        

    def log_final_metrics(self, topo_type):
        thresholds = np.linspace(-1, 1, self.num_thresholds)

        recall = self.averages[topo_type + '_recall'].reset()
        precision = self.averages[topo_type + '_precision'].reset()
        missed = self.averages[topo_type + '_missed'].reset()
        falsepositives = self.averages[topo_type + '_falsepositives'].reset()
        true_positives_and_negatives = self.averages[topo_type + '_true_positives_and_negatives'].reset()
        incorrect_and_falsepositive = self.averages[topo_type + '_incorrect_and_falsepositive'].reset()

        label_indices = [0, len(thresholds) // 2, -1]
        fig_precision_recall = plot_tradeoff(recall, precision, thresholds, label_indices, 'Recall', 'Precision', ' (' + topo_type + ')')
        fig_missed_spurious = plot_tradeoff(missed, falsepositives, thresholds, label_indices, 'Missed', 'False Positive', ' (' + topo_type + ')')

        fig_all = plot_multiple_metrics({
        #'True Negatives': truenegatives,
        #'False Positives': falsepositives,
        'Correct (all)': true_positives_and_negatives,
        'Missed': missed,
        #'Incorrect (Matched)': incorrect,
        'Incorrect or False Positive (all)': incorrect_and_falsepositive,
        }, thresholds, 'Metrics vs. Threshold (' + topo_type + ')')

        fig_precrecall_flat = plot_multiple_metrics({
            'precision': precision,
        'recall': recall}, thresholds, 'Precision & Recall vs. Threshold (' + topo_type + ')')

        self.logger.experiment.add_figure('metric_plots/' + topo_type, fig_all, self.current_epoch)
        self.logger.experiment.add_figure('precision_recall_flat/' + topo_type, fig_precrecall_flat, self.current_epoch)
        self.logger.experiment.add_figure('precision_recall/' + topo_type, fig_precision_recall, self.current_epoch)
        self.logger.experiment.add_figure('missed_falsepositive/' + topo_type, fig_missed_spurious, self.current_epoch)
    

    
    def log_metrics_legacy(self, data, orig_emb, var_emb, topo_type):
        orig_emb_match = orig_emb[getattr(data, topo_type + '_matches')[0]]

        num_batches = count_batches(data)
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
        self.log('accuracy/' + topo_type, acc, batch_size = count_batches(data))
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