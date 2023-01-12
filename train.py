import pytorch_lightning as pl
from argparse import ArgumentParser
from brepmatching.matching_model import MatchingModel
from pytorch_lightning.loggers import TensorBoardLogger
from brepmatching.data import BRepMatchingDataModule
from torch_geometric.loader import DataLoader
import torch
import sys
import os


def fix_file_descriptors():
    # When run in a tmux environment, the file_descriptor strategy can run out
    # of file handles quickly unless you reset the file handle limit
    if sys.platform == "linux" or sys.platform == "linux2":
        import resource
        from torch.multiprocessing import set_sharing_strategy
        set_sharing_strategy("file_descriptor")
        resource.setrlimit(resource.RLIMIT_NOFILE, (100_000, 100_000))


if __name__ == '__main__':
    fix_file_descriptors()
    parser = ArgumentParser(allow_abbrev=False, conflict_handler='resolve')

    parser.add_argument('--tensorboard_path', type=str, default='.')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--name', type=str, default='unnamed')
    parser.add_argument('--resume_version', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--no_test', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--override_args', action='store_true')

    parser = pl.Trainer.add_argparse_args(parser)
    parser = BRepMatchingDataModule.add_argparse_args(parser)
    parser = MatchingModel.add_argparse_args(parser)

    args = parser.parse_args()

    logger = TensorBoardLogger(
        args.tensorboard_path,
        name=args.name,
        default_hp_metric = False,
        version=args.resume_version
    )
    logger.log_hyperparams(args)

    if args.resume_version is not None:
        last_ckpt = os.path.join(
            logger.experiment.log_dir,
            'checkpoints',
            'last.ckpt'
        )

        if not os.path.exists(last_ckpt):
            print(f'No last checkpoint found for version_{args.resume_version}.')
            print(f'Tried {last_ckpt}')
            exit()
        args.checkpoint_path = last_ckpt
        args.resume_from_checkpoint = last_ckpt
    
    data = BRepMatchingDataModule.from_argparse_args(args)

    if args.checkpoint_path is None:
        model = MatchingModel.from_argparse_args(args)
    elif args.override_args:
        model = MatchingModel.from_argparse_args(args)
        model.load_state_dict(torch.load(args.checkpoint_path)['state_dict'])
    else:
        model = MatchingModel.load_from_checkpoint(args.checkpoint_path)
    callbacks = model.get_callbacks()

    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks = callbacks)

    if not args.no_train:
        trainer.fit(model, data)
    
    if not args.no_test:
        if args.no_train:
            if args.checkpoint_path is None:
                print('Testing from initialization.')
            else:
                print(f'Testing from {args.checkpoint_path}')
            if args.validate:
                results = trainer.validate(model, datamodule=data)
            else:
                results = trainer.test(model, datamodule=data)
        else:
            ckpt = trainer.checkpoint_callback.best_model_path
            print(f'Testing from {ckpt}')
            if args.validate:
                results = trainer.validate(datamodule=data)
            else:
                results = trainer.test(datamodule=data)
