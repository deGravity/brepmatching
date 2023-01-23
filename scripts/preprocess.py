import sys
from brepmatching.data import ParallelPreprocessor
from tqdm.contrib.concurrent import thread_map, process_map
from tqdm import tqdm
import torch

zip_path = sys.argv[1]
cache_path = sys.argv[2]

processor = ParallelPreprocessor(
    zip_path=zip_path,
    cache_path=cache_path,
    use_hq=True,
    quality=0.001,
    normalize=True,
    compute_intersection=True
)

dataset = []
for d in tqdm(processor):
    dataset.append(d)

# dataset = thread_map(lambda x: processor.process(x),
                      # range(len(processor)),
                      # max_workers=20,
                      # chunksize=100)

torch.save(dataset, cache_path)

