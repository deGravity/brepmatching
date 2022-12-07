import pytorch_lightning as pl
from argparse import ArgumentParser
from brepmatching.matching_model import MatchingModel
from pytorch_lightning.loggers import TensorBoardLogger
from brepmatching.data import BRepMatchingDataModule
from torch_geometric.loader import DataLoader


if __name__ == '__main__':
    parser = ArgumentParser(allow_abbrev=False, conflict_handler='resolve')

    parser.add_argument('--tensorboard_path', type=str, default='.')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--checkpoint_path2', type=str, default=None)
    parser.add_argument('--name', type=str, default='unnamed')
    parser.add_argument('--resume_version', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--no_test', action='store_true')

    parser = pl.Trainer.add_argparse_args(parser)
    parser = BRepMatchingDataModule.add_argparse_args(parser)
    parser = MatchingModel.add_argparse_args(parser)

    args = parser.parse_args()

    data = BRepMatchingDataModule.from_argparse_args(args)
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
    trainer.fit(model, data)