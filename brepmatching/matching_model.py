import pytorch_lightning as pl
from brepmatching.models import PairEmbedder
import torch
from torch.nn import CrossEntropyLoss, LogSoftmax
from torchmetrics import MeanMetric
import numpy as np

#from torch.profiler import profile, record_function, ProfilerActivity



class MatchingModel(pl.LightningModule):

    def __init__(self,
        f_in_width: int = 62,
        l_in_width: int = 38,
        e_in_width: int = 72,
        v_in_width: int = 3,
        sbgcn_size: int = 64,
        fflayers: int = 6,
        batch_norm: bool = False,

        #use_uvnet_features: bool = False,
        temperature: float = 100.0, #temperature normalization factor for contrastive softmax
        num_negative: int = 5
        
        ):
        super().__init__()

        self.pair_embedder = PairEmbedder(f_in_width, l_in_width, e_in_width, v_in_width, sbgcn_size, fflayers, batch_norm=batch_norm)
        self.temperature = temperature
        self.num_negative = num_negative

        self.loss = CrossEntropyLoss()
        self.softmax = LogSoftmax(dim=1)
        self.batch_norm = batch_norm

        # self.faces_accuracy = MeanMetric()
        # self.edges_accuracy = MeanMetric()
        # self.vertices_accuracy = MeanMetric()
        # self.accuracy = MeanMetric()

        self.save_hyperparameters()
    

    def forward(self, data):
        return self.pair_embedder(data)

    def sample_matches(self, data, topo_type, device='cuda'):
        with torch.no_grad():
            num_batches = self.count_batches(data)
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

    
    def count_batches(self, data):
        return data.left_faces_batch[-1]+1 #Todo: is there a better way to count batches?
    

    def training_step(self, data, batch_idx):
        face_allperms, faces_match_mask = self.sample_matches(data, 'faces', device=data.left_faces.device)
        edge_allperms, edges_match_mask = self.sample_matches(data, 'edges', device=data.left_faces.device)
        vert_allperms, verts_match_mask = self.sample_matches(data, 'vertices', device=data.left_faces.device)
        (f_orig, e_orig, v_orig), (f_var, e_var, v_var) = self(data)
        f_loss = self.compute_loss(face_allperms, data, f_orig, f_var, 'faces', faces_match_mask)
        e_loss = self.compute_loss(edge_allperms, data, e_orig, e_var, 'edges', edges_match_mask)
        v_loss = self.compute_loss(vert_allperms, data, v_orig, v_var, 'vertices', verts_match_mask)
        loss = f_loss + e_loss + v_loss

        batch_size=self.count_batches(data)
        self.log('train_loss/step', loss, on_step=True, on_epoch=False, batch_size=batch_size)
        self.log('train_loss/epoch', loss, on_step=False, on_epoch=True, batch_size=batch_size)
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
        self.log('val_loss', loss, batch_size=self.count_batches(data))

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
        self.log('test_loss', loss, batch_size=self.count_batches(data))


    def allscores(self, data, orig_emb, var_emb, topo_type):
        num_batches = self.count_batches(data)
        batch_left_inds = []
        batch_right_inds = []
        batch_left_feats = []
        batch_right_feats = []
        for j in range(num_batches):
            inds_l = (getattr(data, 'left_'+topo_type+'_batch') == j).nonzero().flatten()
            inds_r = (getattr(data, 'right_'+topo_type+'_batch') == j).nonzero().flatten()
            batch_left_inds.append(inds_l)
            batch_right_inds.append(inds_r)
            batch_left_feats.append(orig_emb[inds_l])
            batch_right_feats.append(var_emb[inds_r])

        similarity_adj = []
        for b in range(num_batches):
            sims = batch_left_feats[b] @ batch_right_feats[b].T
            similarity_adj.append(sims)

        return similarity_adj, batch_left_inds, batch_right_inds
    

    def greedy_matching(self, adjacency_matrices):
        all_matches = []
        all_matching_scores = []
        for adj in adjacency_matrices:
            indices = np.indices(adj.shape)
            coords = indices.transpose([1,2,0]).reshape([-1, 2])
            flattened_adj = adj.flatten()
            sorted_inds = np.argsort(flattened_adj)[::-1]
            sorted_coords = coords[sorted_inds]

            visited_left = np.zeros(adj.shape[0], dtype=np.int8)
            visited_right = np.zeros(adj.shape[1], dtype=np.int8)
            matches = []
            matching_scores = []
            for j in range(sorted_coords.shape[0]):
                coord = sorted_coords[j]
                if visited_left[coord[0]] or visited_right[coord[1]]:
                    continue
                matches.append(coord)
                matching_scores.append(flattened_adj[j])
                visited_left[coord[0]] = 1
                visited_right[coord[1]] = 1
            matches = np.stack(matches)
            matching_scores = np.array(matching_scores)
            all_matches.append(matches)
            all_matching_scores.append(matching_scores)
        return all_matches, all_matching_scores
            
    
    def log_metrics(self, data, orig_emb, var_emb, topo_type):
        scores, batch_left_inds, batch_right_inds = self.allscores(data, orig_emb, var_emb, topo_type)
        scores = list(map(lambda t: t.cpu().numpy(), scores))
        batch_left_inds = list(map(lambda t: t.cpu().numpy(), batch_left_inds))
        batch_right_inds = list(map(lambda t: t.cpu().numpy(), batch_right_inds))
        greedy_matches_all, greedy_scores_all = self.greedy_matching(scores)

        thresholds = [-np.inf] #TODO: choose multiple thresholds
        for threshold in thresholds:
            all_greedy_matches_global = []
            for b in range(len(greedy_matches_all)):
                greedy_matches = greedy_matches_all[b]
                greedy_matches_threshold = [greedy_matches[j] for j in range(len(greedy_matches)) if greedy_scores_all[b][j] > threshold]

                #TODO: global bipartite matching

                #TODO: actual metrics (missing/spurious matches, edge-level metrics)
                greedy_matches_global = [[batch_left_inds[b][match[0]], batch_right_inds[b][match[1]]] for match in greedy_matches]
                all_greedy_matches_global += greedy_matches_global
            
            #temporarily evaluate by computing accuracy of known matches as in legacy version
            num_matches = getattr(data, topo_type + '_matches').shape[1]
            topo2match = np.full(getattr(data, 'left_' + topo_type).shape[0:1],-1)
            topo2match[getattr(data, topo_type + '_matches')[0]] = np.arange(num_matches)
            all_greedy_matches_global = [match for match in all_greedy_matches_global if topo2match[match[0]] >= 0]
            all_greedy_matches_global = np.array(all_greedy_matches_global).T
            #unordered matches -> left topo index -> left original match index (if exists)
            right_matches = np.full((num_matches,), -1, dtype=np.int)
            right_matches[topo2match[all_greedy_matches_global[0]]] = all_greedy_matches_global[1]

            gt_matches = getattr(data, topo_type + '_matches')[1].cpu().numpy()
            acc = (right_matches == gt_matches).sum() / len(gt_matches)
            self.log('accuracy/' + topo_type, acc, batch_size = self.count_batches(data))


    
    def log_metrics_legacy(self, data, orig_emb, var_emb, topo_type):
        orig_emb_match = orig_emb[getattr(data, topo_type + '_matches')[0]]

        num_batches = self.count_batches(data)
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
        self.log('accuracy/' + topo_type, acc, batch_size = self.count_batches(data))
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
