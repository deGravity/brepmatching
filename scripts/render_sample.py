from argparse import ArgumentParser
from brepmatching.data import BRepMatchingDataset
from brepmatching.visualization import render_predictions, show_image
import os
import numpy as np
from tqdm import tqdm

def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--outdir', type=str)
    parser.add_argument('-n', type=int, default=100)

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f'Could not find {args.dataset}.')
        exit()
    

    os.makedirs(args.outdir, exist_ok=True)
    ds_dir = os.path.dirname(args.dataset)
    ds_name = os.path.basename(args.dataset)
    cache_name = '.'.join(ds_name.split('.')[:-1]) + '.pt'
    cache_path = os.path.join(ds_dir, cache_name)

    ds = BRepMatchingDataset(args.dataset, cache_path, test_size=0.0, val_size=0.0)

    n = args.n
    if n < 0:
        n = len(ds)
    if n > len(ds):
        n = len(ds)

    to_render = np.random.choice(len(ds), n, replace=False)


    for i in tqdm(to_render, "Rendering Images"):
        data = ds[i]
        idx = ds.original_index[i]
        image = show_image(render_predictions(data, data.faces_matches))
        image.save(os.path.join(args.outdir, f'{idx}.png'))

if __name__ == '__main__':
    main()