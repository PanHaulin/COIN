import json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
import time

class MonitorCheckpointer(ModelCheckpoint):
    '''
    具有监控指标、保存topk和保存last功能的checkpointer
    '''
    def __init__(
        self,
        args: Namespace,
        logdir: Union[str, Path] = Path("trained_models"),
        frequency: int = 1,
        monitor: Optional[str] = None,
        mode: str = "min",
        save_last: Optional[bool] = None,
        save_top_k: int = 1,
    ):
        """Custom checkpointer callback that stores checkpoints in an easier to access way.

        Args:
            args (Namespace): namespace object containing at least an attribute name.
            logdir (Union[str, Path], optional): base directory to store checkpoints.
                Defaults to "trained_models".
            frequency (int, optional): number of epochs between each checkpoint. Defaults to 1.
            keep_previous_checkpoints (bool, optional): whether to keep previous checkpoints or not.
                Defaults to False.
        """

        super().__init__(
            dirpath=logdir,
            filename="{epoch:02d},{val_acc1:.4f},{val_total_loss:.4f}",
            monitor=monitor,
            mode = mode,
            save_last=save_last,
            save_top_k=save_top_k,
        )

        self.args = args
        self.logdir = Path(logdir)
        self.frequency = frequency
        self.monitor = monitor
        # self.keep_previous_checkpoints = keep_previous_checkpoints

    @staticmethod
    def add_checkpointer_args(parent_parser: ArgumentParser):
        """Adds user-required arguments to a parser.

        Args:
            parent_parser (ArgumentParser): parser to add new args to.
        """

        parser = parent_parser.add_argument_group("checkpointer")
        parser.add_argument("--checkpoints_savedir", default="trained_models", type=str)
        parser.add_argument("--checkpoint_frequency", default=1, type=int)
        parser.add_argument("--checkpoint_monitor", default='val_acc1', type=str)
        parser.add_argument("--checkpoint_mode", default="max", type=str)
        parser.add_argument("--checkpoint_save_last", action="store_true")
        parser.add_argument("--checkpoint_save_topk", default=1, type=int)
        parser.add_argument("--resume", action="store_true")
        parser.add_argument('--resume_dir', type=str, default=None)
        parser.add_argument('--resume_checkpoint_filename', type=str, default=None)

        return parent_parser

    def initial_setup(self, trainer: pl.Trainer):
        """Creates the directories and does the initial setup needed.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        # # create logging dirs
        if trainer.is_global_zero:
            os.makedirs(self.logdir, exist_ok=True)
        pass

    def save_args(self, trainer: pl.Trainer):
        """Stores arguments into a json file.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        if trainer.is_global_zero:
            args = vars(self.args)
            json_path = self.logdir / "args.json"
            json.dump(args, open(json_path, "w"), default=lambda o: "<not serializable>")

    def on_train_start(self, trainer: pl.Trainer, _):
        """Executes initial setup and saves arguments.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        self.initial_setup(trainer) # 设置path和ckpt_placeholder
        self.save_args(trainer)
        # 调用父类hook
        super().on_train_start(trainer, _)
    
    @staticmethod
    def get_instance(args):
        """
        生成实例
        """
        return MonitorCheckpointer(
            args,
            logdir=args.logdir,
            frequency=args.checkpoint_frequency,
            monitor=args.checkpoint_monitor,
            mode=args.checkpoint_mode,
            save_last=args.checkpoint_save_last,
            save_top_k=args.checkpoint_save_topk
        )
