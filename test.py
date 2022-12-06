from brepmatching.data import BRepMatchingDataset
from brepmatching.matching_model import MatchingModel
from torch_geometric.loader import DataLoader


if __name__ == '__main__':
    ds = BRepMatchingDataset('data/example_dataset.zip', 'tmp/example_dataset_cache.pt', debug=True)
    dl = DataLoader(ds, batch_size=10, shuffle=True, follow_batch=['left_vertices','right_vertices','left_edges', 'right_edges','left_faces','right_faces', 'face_matches', 'edge_matches', 'vertex_matches'])
    matcher = MatchingModel()
    batch = next(iter(dl))
    matcher(batch)