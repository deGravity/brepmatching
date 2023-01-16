from contextlib import contextmanager
from automate import HetData
from torch import is_tensor
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional
from pathlib import Path
import os

TOPO_KINDS: list[tuple[str, str, str]] = [
  ("faces", "face", "f"),
  # ("loops", "loop", "l"),
  ("edges", "edge", "e"),
  ("vertices", "vertex", "v")
]


def zip_hetdata(left, right):
    common_keys = set(left.keys).intersection(right.keys)
    data = HetData()
    for k in common_keys:
        # A bit of a hack to remove non-batchable items
        # The better place to fix this would be in Automate itself
        if not is_tensor(left[k]) or not is_tensor(right[k]) or len(left[k].shape) == 0 or len(right[k].shape) == 0:
            continue
        data['left_' + k] = left[k]
        data['right_' + k] = right[k]
    for k,v in left.__edge_sets__.items():
        data.__edge_sets__['left_' + k] = ['left_' + name if isinstance(name, str) else name  for name in v]
    for name in left.__node_sets__:
        data.__node_sets__.add('left_' + name)
    for k,v in right.__edge_sets__.items():
        data.__edge_sets__['right_' + k] = ['right_' + name if isinstance(name, str) else name  for name in v]
    for name in right.__node_sets__:
        data.__node_sets__.add('right_' + name)
    return data

def unzip_hetdata(data):
    left_keys = [k for k in data.keys if k.startswith('left_')]
    right_keys = [k for k in data.keys if k.startswith('right_')]
    left = HetData()
    right = HetData()
    for k in left_keys:
        left[k[5:]] = data[k]
    for k in right_keys:
        right[k[6:]] = data[k]
    for k,v in data.__edge_sets__.items():
        if k.startswith('left_'):
            left.__edge_sets__[k[5:]] = [name[5:] if isinstance(name, str) else name for name in v]
        elif k.startswith('right_'):
            right.__edge_sets__[k[6:]] = [name[6:] if isinstance(name, str) else name for name in v]
    for name in data.__node_sets__:
        if name.startswith('left_'):
            left.__node_sets__.add(name[5:])
        elif name.startswith('right_'):
            right.__node_sets__.add(name[6:])
    return left, right

def zip_apply(network, zipped_data):
    left, right = unzip_hetdata(zipped_data)
    return network(left), network(right)

def zip_apply_2(network1, network2, zipped_data):
    left, right = unzip_hetdata(zipped_data)
    return network1(left), network2(right)


def count_batches(data: HetData) -> int:
    #Todo: is there a better way to count batches?
    return int(max(data.left_faces_batch[-1].item(), data.right_faces_batch[-1].item())) + 1

###### MATCHING ######

def greedy_matching(adjacency_matrices):
    """
    Performs greedy matching based on descending similarity score
    Parameters:
    - adjacency_matrices: Array of similarity matrices, where the i,jth element specifies the similarity between the ith topology on the left and the jth topology on the right

    Returns:
    - all_matches: array of matches, each of which is a n x 2 numpy arrays containing the matching topo indices for a pair of parts
    - all_matching_scores: array of matching scores, each of which is an n-length numpy array containing the similarity scores for each corresponding matching (see above)
    """

    all_matches = []
    all_matching_scores = []
    for adj in adjacency_matrices:
        if adj.shape[0] == 0 or adj.shape[1] == 0:
            all_matches.append([])
            all_matching_scores.append([])
            continue
        indices = np.indices(adj.shape)
        coords = indices.transpose([1,2,0]).reshape([-1, 2])
        flattened_adj = adj.flatten()
        sorted_inds = np.argsort(flattened_adj)[::-1]
        sorted_coords = coords[sorted_inds]

        visited_left = np.zeros(adj.shape[0], dtype=np.int8)
        visited_right = np.zeros(adj.shape[1], dtype=np.int8)
        matches = []
        matching_scores = []
        #add matches in descending order of score, greedily
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

def greedy_match_2(scores: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given the scores matrix `scores`, greedily match based on descending scores.

    Complexity `O(n * GPU(n ^ 2))`

    Returns:
    - `all_matches`: 2 x m tensor
    - `all_scores`: m tensor
    """
    if mask == None:
        mask = torch.ones_like(scores, dtype=torch.bool, device=scores.device)
    else:
        mask = mask.clone()

    all_matches = []
    all_scores = []

    while torch.any(mask):
        mx = scores[mask].max()

        # take first maximum score
        l, r = (int(x.item()) for x in (scores == mx).logical_and(mask).nonzero()[0])

        # add (l, r) to matches
        all_matches.append([l, r])
        all_scores.append(mx)

        # do not consider l and r again
        mask[l, :] = False
        mask[:, r] = False

    return torch.tensor(all_matches, dtype=torch.long, device=scores.device).T, \
        torch.tensor(all_scores, device=scores.device)

def separate_batched_adj_mtx(mtx: torch.Tensor,
                             left_topo_batches: torch.Tensor,
                             right_topo_batches: torch.Tensor) -> list[torch.Tensor]:
    """
    Given a batched adjacency matrix, return a list of unbatched adjacency matrices.

    Assume a batch is contiguous.
    The returned tensors are views of the original.
    """

    num_batches = int(max(left_topo_batches[-1].item(), right_topo_batches[-1].item())) + 1
    mtx_list = []
    l_offset = 0
    r_offset = 0
    for b in range(num_batches):
        l_count = (left_topo_batches == b).sum()
        r_count = (right_topo_batches == b).sum()
        mtx_list.append(mtx[l_offset:l_offset + l_count, r_offset:r_offset + r_count])
        l_offset += l_count
        r_offset += r_count
    return mtx_list


def separate_batched_matches(matches: torch.Tensor,
                             left_topo_batches: torch.Tensor,
                             right_topo_batches: torch.Tensor) -> list[torch.Tensor]:
    """
    given a 2xn tensor of matches, and the batches tensor of the left and right nodes into which the matches index,
    return a list of b matches, with local indices within each instance
    """
    match_list = []
    num_batches = int(max(left_topo_batches[-1].item(), right_topo_batches[-1].item())) + 1
    match_batches = left_topo_batches[matches[0]]
    left_batch_counts = [(left_topo_batches == b).sum() for b in range(num_batches)]
    right_batch_counts = [(right_topo_batches == b).sum() for b in range(num_batches)]
    left_batch_offsets = []
    right_batch_offsets = []
    device = matches.device
    offset = torch.tensor(0, device=device)
    for size in left_batch_counts:
        left_batch_offsets.append(offset.clone())
        offset += size
    offset = torch.tensor(0, device=device)
    for size in right_batch_counts:
        right_batch_offsets.append(offset.clone())
        offset += size
    
    for b in range(num_batches):
        filtered_matches = matches[:, match_batches == b]
        filtered_matches[0] -= left_batch_offsets[b]
        filtered_matches[1] -= right_batch_offsets[b]
        assert(filtered_matches[0].numel() == 0 or (0 <= filtered_matches[0].min() and filtered_matches[0].max() < left_batch_counts[b]))
        assert(filtered_matches[1].numel() == 0 or (0 <= filtered_matches[1].min() and filtered_matches[1].max() < right_batch_counts[b]))
        match_list.append(filtered_matches)
    return match_list


def batch_matches(matches, left_topo_batches, right_topo_batches):
    """
    Batch together matches, which are a list of lists of match pairs (NOT a list of 2xN tensors!)
    input are numpy arrays
    """
    num_batches = len(matches)
    batch_left_inds = [(left_topo_batches == j).nonzero()[0] for j in range(num_batches)]
    batch_right_inds = [(right_topo_batches == j).nonzero()[0] for j in range(num_batches)]
    return sum([[[batch_left_inds[b][match[0]], batch_right_inds[b][match[1]]] for match in matches[b]] for b in range(num_batches)], [])

###### ADJACENCY ######

def construct_adjacency_matrix(edge_indices: torch.Tensor, n_a, n_b) -> torch.Tensor:
    res = torch.zeros((n_a, n_b), dtype=torch.bool, device=edge_indices.device)
    res[edge_indices[0], edge_indices[1]] = True
    return res

def construct_adjacency_list(edge_indices: torch.Tensor, n_a, n_b) -> tuple[list[list[int]], list[list[int]]]:
    res_ab: list[list[int]] = [[] for _ in range(n_a)]
    res_ba: list[list[int]] = [[] for _ in range(n_b)]
    n_e = edge_indices.shape[1]
    for i in range(n_e):
        u = int(edge_indices[0, i].item())
        v = int(edge_indices[1, i].item())
        res_ab[u].append(v)
        res_ba[v].append(u)
    return res_ab, res_ba

def combine_adjacency_list(a2b: list[list[int]], b2c: list[list[int]]) -> list[list[int]]:
    a2c: list[list[int]] = [[] for _ in range(len(a2b))]
    for i, bs in enumerate(a2b):
        for b in bs:
            a2c[i].extend(b2c[b])
        # unique
        a2c[i] = list(set(a2c[i]))
    return a2c

# TODO might not be the most efficient way
def matmul_bool(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Complexity O(n * GPU(n ^ 2))
    """
    p, q1 = a.shape
    q2, r = b.shape
    assert(q1 == q2)
    res = torch.empty((p, r), dtype=torch.bool, device=a.device)
    for j in range(r):
        res[:, j] = a.logical_and(b.T[j]).any(dim=-1)
    return res

def matmul_(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a @ b).clamp(None, 1)


def precompute_adjacency(data: HetData) -> dict[str, dict[str, list[list[int]]]]:
    n_faces = data.faces.shape[0]
    n_loops = data.loops.shape[0]
    n_edges = data.edges.shape[0]
    n_vertices = data.vertices.shape[0]

    f2l, l2f = construct_adjacency_list(data.face_to_loop, n_faces, n_loops)
    l2e, e2l = construct_adjacency_list(data.loop_to_edge, n_loops, n_edges)
    e2v, v2e = construct_adjacency_list(data.edge_to_vertex, n_edges, n_vertices)

    f2e = combine_adjacency_list(f2l, l2e)
    e2f = combine_adjacency_list(e2l, l2f)

    f2v = combine_adjacency_list(f2e, e2v)
    v2f = combine_adjacency_list(v2e, e2f)

    return {
        "f": {"f": [[] for _ in range(n_faces)], "e": f2e, "v": f2v },
        "e": {"f": e2f, "e": [[] for _ in range(n_edges)], "v": e2v },
        "v": {"f": v2f, "e": e2v, "v": [[] for _ in range(n_vertices)]}
    }

def add_match_to_frontier(u: int, v: int, ka: str,
                          adj_data_l: dict[str, dict[str, list[list[int]]]],
                          adj_data_r: dict[str, dict[str, list[list[int]]]],
                          adj_l: dict[str, torch.Tensor],
                          adj_r: dict[str, torch.Tensor]) -> None:
    for _, _, kb in TOPO_KINDS:
        adj_l[kb][adj_data_l[ka][kb][u]] = True
        adj_r[kb][adj_data_r[ka][kb][v]] = True

def propagate_adjacency(data: HetData,
                        adj_left: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                        adj_right: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                       ) -> dict[str, torch.Tensor]:
    device = data.left_faces.device

    n_faces_left = data.left_faces.shape[0]
    n_edges_left = data.left_edges.shape[0]
    n_vertices_left = data.left_vertices.shape[0]

    match_faces_left = torch.zeros((1, n_faces_left), dtype=torch.bool, device=device)
    match_faces_left[0, data.cur_faces_matches[0]] = True

    match_edges_left = torch.zeros((1, n_edges_left), dtype=torch.bool, device=device)
    match_edges_left[0, data.cur_edges_matches[0]] = True

    match_vertices_left = torch.zeros((1, n_vertices_left), dtype=torch.bool, device=device)
    match_vertices_left[0, data.cur_vertices_matches[0]] = True

    left_f2e, left_f2v, left_e2v = adj_left

    adj_faces_left = matmul_bool(match_edges_left, left_f2e.T).logical_or(matmul_bool(match_vertices_left, left_f2v.T))
    adj_edges_left = matmul_bool(match_faces_left, left_f2e).logical_or(matmul_bool(match_vertices_left, left_e2v.T))
    adj_vertices_left = matmul_bool(match_faces_left, left_f2v).logical_or(matmul_bool(match_edges_left, left_e2v))

    # TODO: Fix this massive copy and paste
    n_faces_right = data.right_faces.shape[0]
    n_edges_right = data.right_edges.shape[0]
    n_vertices_right = data.right_vertices.shape[0]

    match_faces_right = torch.zeros((1, n_faces_right), dtype=torch.bool, device=device)
    match_faces_right[0, data.cur_faces_matches[1]] = True

    match_edges_right = torch.zeros((1, n_edges_right), dtype=torch.bool, device=device)
    match_edges_right[0, data.cur_edges_matches[1]] = True

    match_vertices_right = torch.zeros((1, n_vertices_right), dtype=torch.bool, device=device)
    match_vertices_right[0, data.cur_vertices_matches[1]] = True

    right_f2e, right_f2v, right_e2v = adj_right

    adj_faces_right = matmul_bool(match_edges_right, right_f2e.T).logical_or(matmul_bool(match_vertices_right, right_f2v.T))
    adj_edges_right = matmul_bool(match_faces_right, right_f2e).logical_or(matmul_bool(match_vertices_right, right_e2v.T))
    adj_vertices_right = matmul_bool(match_faces_right, right_f2v).logical_or(matmul_bool(match_edges_right, right_e2v))

    return {
        "f": matmul_bool(adj_faces_left.T, adj_faces_right),
        "e": matmul_bool(adj_edges_left.T, adj_edges_right),
        "v": matmul_bool(adj_vertices_left.T, adj_vertices_right)
    }

###### METRICS ######

NUM_METRICS = 7
METRIC_COLS = ["true_pos", "true_neg", "missed", "incorrect", "false_pos", # relative to nr
               "precision", "recall"]

def compute_metrics_impl(matches: torch.Tensor,
                         gt_matches: torch.Tensor,
                         n_topos_left: int,
                         n_topos_right: int) -> np.ndarray:
    device = matches.device

    pred = torch.full((n_topos_right, ), -1, device=device)
    pred[matches[1]] = matches[0]

    gt = torch.full((n_topos_right, ), -1, device=device)
    gt[gt_matches[1]] = gt_matches[0]

    num_gt_matched = int((gt >= 0).sum().item())
    num_gt_unmatched = n_topos_right - num_gt_matched

    num_matched = int((pred >= 0).sum().item())

    correct_mask = (pred == gt)
    num_correct = int(correct_mask.sum().item())
    num_true_pos = int(correct_mask.logical_and(pred >= 0).sum().item())
    num_true_neg = num_correct - num_true_pos

    incorrect_mask = (pred != gt)
    num_incorrect = int(incorrect_mask.sum())
    num_false_pos = int((gt[pred >= 0] == -1).sum())
    num_missed = int((gt[pred == -1] >= 0).sum())
    num_wrong_pos = num_incorrect - num_false_pos - num_missed

    true_pos = (num_true_pos / n_topos_right) if n_topos_right > 0 else 0.0
    true_neg = (num_true_neg / n_topos_right) if n_topos_right > 0 else 1.0
    missed = (num_missed / n_topos_right) if n_topos_right > 0 else 0.0
    incorrect = (num_wrong_pos / n_topos_right) if n_topos_right > 0 else 0.0
    false_pos = (num_false_pos / n_topos_right) if n_topos_right > 0 else 0.0

    precision = (num_true_pos / num_matched) if num_matched > 0 else 1.0
    recall = (num_true_pos / num_gt_matched) if num_gt_matched > 0 else 1.0

    return np.array([true_pos, true_neg, missed, incorrect, false_pos, precision, recall])

def compute_metrics_from_matches(data: HetData, kinds: str, matches: torch.Tensor) -> np.ndarray:
    gt_matches = data[f"{kinds}_matches"]       # assume non-empty

    batch_left = data[f"left_{kinds}_batch"]
    batch_right = data[f"right_{kinds}_batch"]

    cur_matches_unbatched = separate_batched_matches(matches, batch_left, batch_right)
    gt_matches_unbatched = separate_batched_matches(gt_matches, batch_left, batch_right)

    n_batches = len(cur_matches_unbatched)

    metrics = np.zeros(NUM_METRICS)

    for b in range(n_batches):
        n_topos_left = int((batch_left == b).sum().item())
        n_topos_right = int((batch_right == b).sum().item())

        cur_matches_b = cur_matches_unbatched[b]
        gt_matches_b = gt_matches_unbatched[b]

        cur_metrics = compute_metrics_impl(
            cur_matches_b, gt_matches_b, n_topos_left, n_topos_right)
        
        metrics += cur_metrics
    
    metrics /= n_batches

    return metrics

###### PLOTTING ######

def plot_metric(metric, thresholds, name):
    fig = Figure(figsize=(8, 8))
    ax = fig.add_subplot()
    ax.plot(thresholds, metric)
    ax.set_title(name + ' vs threshold')
    ax.set_xlabel('Threshold')
    ax.set_ylabel(name)
    ax.set_ylim(-0.1, 1.1)
    ax.grid()
    return fig

def plot_the_fives(true_pos: np.ndarray,
                   true_neg: np.ndarray,
                   missed: np.ndarray,
                   incorrect: np.ndarray,
                   false_pos: np.ndarray,
                   thresholds: np.ndarray,
                   title: str,
                   ax: plt.Axes = None) -> Figure:
    fig = None
    if ax is None:
        fig = Figure(figsize=(8, 8))
        ax = fig.add_subplot()
    ax.stackplot(thresholds, false_pos, incorrect, missed, true_neg, true_pos,
                 labels=["False Positive", "Incorrect", "Missed", "True Negative", "True Positive"],
                 colors=["#BA5050", "#D4756C", "#D6CFB8", "#61B5CF", "#468CB8"])
    ax.legend(loc="upper left")
    ax.set_xlabel("Threshold")
    ax.set_title(title)
    ax.set_ylim(-0.1, 1.1)
    ax.grid()
    
    return fig
    
    
def plot_multiple_metrics(metrics: dict[str, np.ndarray], 
                          thresholds: np.ndarray,
                          title: str):
    fig = Figure(figsize=(8, 8))
    ax = fig.add_subplot()
    for j, key in enumerate(metrics):
        if j == 0:
            color = '#0000ff'
        elif j == 1:
            color = '#e08c24'
        elif j == 2:
            color = '#ff0000'
        else:
            color = None
        ax.plot(thresholds, metrics[key], label=key, color=color)
    ax.legend()
    ax.set_xlabel('Threshold')
    ax.set_title(title)
    ax.set_ylim(-0.1, 1.1)
    ax.grid()
    return fig

def plot_tradeoff(x, y, values, indices, xname, yname, suffix=''):
    fig = Figure(figsize=(8, 8))
    ax = fig.add_subplot()
    ax.plot(x, y)

    x_filtered = [x[i] for i in indices]
    y_filtered = [y[i] for i in indices]
    v_filtered = [values[i] for i in indices]
    ax.scatter(x_filtered, y_filtered)
    for xf, yf, vf in zip(x_filtered, y_filtered, v_filtered):
        ax.annotate(str(round(vf,2)), (xf, yf))

    ax.set_title(yname + ' VS ' + xname + suffix)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.grid()
    return fig

class Running_avg:
    def __init__(self, dim):
        self.state = np.zeros(dim)
        self.count = 0.0
    def __call__(self, value, weight):
        self.state += value * weight
        self.count += weight
    def reset(self):
        val = self.state / self.count if self.count > 0 else 0
        self.state[:] = 0
        self.count = 0.0
        return val


def make_containing_dir(filepath):
    """Ensure that the directory part of filepath exists."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

@contextmanager
def safe_open(path, mode):
    """Open a file and create parent directories if necessary."""
    make_containing_dir(path)
    file = open(path, mode)
    try:
        yield file
    finally:
        file.close()

def gdrive_wget(FILEID, FILENAME):
    """Create a wget command to download a Google Drive file from a terminal."""
    return f"wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=" + \
        "$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate" + \
        " 'https://docs.google.com/uc?export=download&id={FILEID}' -O- | sed -rn 's/.*confirm=([0-9A-" + \
        "Za-z_]+).*/\1\n/p')&id={FILEID}\" -O {FILENAME} && rm -rf /tmp/cookies.txt"

def url_to_wget(url, dest):
    """Create a wget command from a Google Drive link."""
    url = url.split('/d/')[-1]
    return gdrive_wget(url, dest)

"""Optional Onshape API helpers"""
try:
    from apikey.onshape import Onshape
    from collections import namedtuple
    
    Element = namedtuple('Element', ['did', 'wid', 'mv', 'eid'])

    api = Onshape(stack='https://cad.onshape.com', creds=str(Path.home().joinpath('.config','onshapecreds.json')), logging=False)
    def dowload_part(did, mv, eid):
        response = api.request(
        method='get', 
        path=f'/api/partstudios/d/{did}/m/{mv}/e/{eid}/parasolid', query={'includeExportIds':True})
        return response.text
    
    def parts_from_url(url):
        parts = url.split('/')
        return Element(parts[-7], parts[-5], parts[-3], parts[-1])
    def dowload_part_from_url(url):
        el= parts_from_url(url)
        return dowload_part(el.did, el.mv, el.eid)
except ModuleNotFoundError as err:
    pass # Dont create these helpers if apikey is not installed

def logsumexp(x, keep_mask=None, add_one=True, dim=1):
    if keep_mask is not None:
        x = x.masked_fill(~keep_mask, -torch.inf)
    if add_one:
        zeros = torch.zeros(x.size(dim - 1), dtype=x.dtype, device=x.device).unsqueeze(
            dim
        )
        x = torch.cat([x, zeros], dim=dim)

    output = torch.logsumexp(x, dim=dim, keepdim=True)
    if keep_mask is not None:
        output = output.masked_fill(~torch.any(keep_mask, dim=dim, keepdim=True), 0)
    return output