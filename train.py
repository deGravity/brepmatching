import pytorch_lightning as pl
from argparse import ArgumentParser
from brepmatching.matching_model import MatchingModel
from pytorch_lightning.loggers import TensorBoardLogger
from brepmatching.data import BRepMatchingDataset
from torch_geometric.loader import DataLoader


if __name__ == '__main__':
    parser = ArgumentParser(allow_abbrev=False, conflict_handler='resolve')

    parser.add_argument('--tensorboard_path', type=str, default='.')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--checkpoint_path2', type=str, default=None)
    parser.add_argument('--name', type=str, default='unnamed')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--resume_version', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--no_test', action='store_true')

    parser.add_argument('--batch_size',type=int, default=16) #TODO: replace this with data module args

    parser = pl.Trainer.add_argparse_args(parser)
    parser = MatchingModel.add_argparse_args(parser)

    args = parser.parse_args()

    ds = BRepMatchingDataset('data/example_dataset.zip', 'tmp/example_dataset_cache.pt', debug=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, follow_batch=['left_vertices','right_vertices','left_edges', 'right_edges','left_faces','right_faces', 'face_matches', 'edge_matches', 'vertex_matches'])

    model = MatchingModel.from_argparse_args(args)
    callbacks = model.get_callbacks()

    logger = TensorBoardLogger(
        args.tensorboard_path,
        name=args.name,
        default_hp_metric = False,
        version=args.resume_version
    )
    logger.log_hyperparams(args)

    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks = callbacks)
    trainer.fit(model, dl)