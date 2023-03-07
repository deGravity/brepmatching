from automate import HetData
from torch import is_tensor
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import torch

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


def count_batches(data):
    return data.left_faces_batch[-1]+1 #Todo: is there a better way to count batches?


def greedy_matching(adjacency_matrices, existing_matches=None):
    """
    Performs greedy matching based on descending similarity score
    Parameters:
    - adjacency_matrices: Array of similarity matrices, where the i,jth element specifies the similarity between the ith topology on the left and the jth topology on the right
    - existing matches: list of existing matches (each 2xN) within each batch
    Returns:
    - all_matches: array of matches, each of which is a n x 2 numpy arrays containing the matching topo indices for a pair of parts
    - all_matching_scores: array of matching scores, each of which is an n-length numpy array containing the similarity scores for each corresponding matching (see above)
    """

    all_matches = []
    all_matching_scores = []
    for b,adj in enumerate(adjacency_matrices):
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
        if existing_matches is not None:
            visited_left[existing_matches[b][0]] = 1
            visited_right[existing_matches[b][1]] = 1

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

def separate_batched_matches(matches, left_topo_batches, right_topo_batches):
    """
    given a 2xn tensor of matches, and the batches tensor of the left and right nodes into which the matches index,
    return a list of b matches, with local indices within each instance
    """
    match_list = []
    num_batches = left_topo_batches[-1] + 1
    match_batches = left_topo_batches[matches[0]]
    left_batch_counts = [(left_topo_batches == b).sum() for b in range(num_batches)]
    right_batch_counts = [(right_topo_batches == b).sum() for b in range(num_batches)]
    left_batch_offsets = []
    right_batch_offsets = []
    offset = torch.tensor(0)
    for size in left_batch_counts:
        left_batch_offsets.append(offset.clone())
        offset += size
    offset = torch.tensor(0)
    for size in right_batch_counts:
        right_batch_offsets.append(offset.clone())
        offset += size
    
    for b in range(num_batches):
        filtered_matches = matches[:, match_batches == b]
        filtered_matches[0] -= left_batch_offsets[b]
        filtered_matches[1] -= right_batch_offsets[b]
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

    num_batches = count_batches(data)
    cur_matches_unbatched = separate_batched_matches(matches, batch_left, batch_right, num_batches)
    gt_matches_unbatched = separate_batched_matches(gt_matches, batch_left, batch_right, num_batches)

    metrics = np.zeros(NUM_METRICS)

    for b in range(num_batches):
        n_topos_left = int((batch_left == b).sum().item())
        n_topos_right = int((batch_right == b).sum().item())

        cur_matches_b = cur_matches_unbatched[b]
        gt_matches_b = gt_matches_unbatched[b]

        cur_metrics = compute_metrics_impl(
            cur_matches_b, gt_matches_b, n_topos_left, n_topos_right)
        
        metrics += cur_metrics
    
    metrics /= num_batches

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