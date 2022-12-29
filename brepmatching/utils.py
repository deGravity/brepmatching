from automate import HetData
from torch import is_tensor
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

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


def greedy_matching(adjacency_matrices):
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


def compute_metrics(data, greedy_matches_all, greedy_scores_all, topo_type, thresholds):
    """
    greedy_matches_all: array of matches, each of which is a n x 2 numpy arrays containing the matching topo indices for a pair of parts
    greedy_scores_all: array of matching scores, each of which is an n-length numpy array containing the similarity scores for each corresponding matching (see above)
    topo_type: (faces | edges | vertices)
    """

    num_batches = count_batches(data)

    batch_left = getattr(data, 'left_'+topo_type+'_batch')
    batch_right = getattr(data, 'right_'+topo_type+'_batch')
    batch_left_inds = [(batch_left == j).nonzero().flatten().cpu().numpy() for j in range(num_batches)]
    batch_right_inds = [(batch_right == j).nonzero().flatten().cpu().numpy() for j in range(num_batches)]

    truenegatives = []
    falsepositives = []
    missed = []
    incorrect = []
    true_positives_and_negatives = []
    incorrect_and_falsepositive = []
    precision = []
    recall = []

    for j,threshold in  enumerate(thresholds):
        all_greedy_matches_global_raw = [] #all matches in all batches (with global indexing) for the current threshold value
        for b in range(len(greedy_matches_all)):
            if len(greedy_scores_all[b]) == 0:
                continue
            greedy_matches = greedy_matches_all[b]
            greedy_matches_threshold = [greedy_matches[j] for j in range(len(greedy_matches)) if greedy_scores_all[b][j] > threshold]

            #TODO: global bipartite matching

            #TODO: actual metrics (missing/spurious matches, edge-level metrics)
            greedy_matches_global = [[batch_left_inds[b][match[0]], batch_right_inds[b][match[1]]] for match in greedy_matches_threshold]
            all_greedy_matches_global_raw += greedy_matches_global
        
        all_greedy_matches_global = np.array(all_greedy_matches_global_raw).T
        
        left_num_topos = getattr(data, 'left_' + topo_type).shape[0]
        right_num_topos = getattr(data, 'right_' + topo_type).shape[0]
        left_gt_matches = getattr(data, topo_type + '_matches')[0].cpu().numpy()
        right_gt_matches = getattr(data, topo_type + '_matches')[1].cpu().numpy()
        left_topo2match = np.full((left_num_topos,),-1)
        right_topo2gtmatch = np.full((right_num_topos,),-1)
        left_topo2match[left_gt_matches] = right_gt_matches
        right_topo2gtmatch[right_gt_matches] = left_gt_matches

        right_topo2match = np.full((right_num_topos,), -1)
        if len(all_greedy_matches_global) > 0:
            right_topo2match[all_greedy_matches_global[1]] = all_greedy_matches_global[0]
        num_gt_matched = (right_topo2gtmatch >= 0).sum()
        num_gt_unmatched = right_num_topos - num_gt_matched
        num_matched = (right_topo2match >= 0).sum()



        matched_mask = (right_topo2match == right_topo2gtmatch)
        num_correct = matched_mask.sum()
        num_truepositive = (matched_mask & (right_topo2match >= 0)).sum()
        num_truenegative = num_correct - num_truepositive

        num_missed = (right_topo2gtmatch[right_topo2match == -1] >= 0).sum()

        incorrect_mask = (right_topo2match != right_topo2gtmatch)
        num_incorrect = incorrect_mask.sum()
        num_false_positive = (right_topo2gtmatch[right_topo2match >= 0] == -1).sum()
        num_wrong_positive = num_incorrect - num_false_positive - num_missed

        #truepositives.append(num_truepositive / num_gt_matched)
        truenegatives.append((num_truenegative / num_gt_unmatched) if num_gt_unmatched > 0 else 0)
        falsepositives.append((num_false_positive / num_gt_unmatched) if num_gt_unmatched > 0 else 0)
        missed.append(num_missed / num_gt_matched)
        incorrect.append(num_wrong_positive / num_gt_matched)
        true_positives_and_negatives.append(num_correct / right_num_topos)
        incorrect_and_falsepositive.append((num_wrong_positive + num_false_positive) / right_num_topos)

        precision.append(num_truepositive / num_matched)
        recall.append(num_truepositive / num_gt_matched)

        if j == 0:
            right2left_matched_accuracy = num_truepositive / num_gt_matched

    return truenegatives, falsepositives, missed,  incorrect, true_positives_and_negatives, incorrect_and_falsepositive, precision, recall, right2left_matched_accuracy

def plot_metric(metric, thresholds, name):
    fig = Figure(figsize=(8, 8))
    ax = fig.add_subplot()
    ax.plot(thresholds, metric)
    ax.set_title(name + ' vs threshold')
    ax.set_xlabel('Threshold')
    ax.set_ylabel(name)
    ax.set_ylim(-0.1, 1.1)
    return fig
    
def plot_multiple_metrics(metrics, thresholds, name):
    fig = Figure(figsize=(8, 8))
    ax = fig.add_subplot()
    for key in metrics:
        ax.plot(thresholds, metrics[key], label=key)
    ax.legend()
    ax.set_xlabel('Threshold')
    ax.set_title(name)
    ax.set_ylim(-0.1, 1.1)
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
    return fig