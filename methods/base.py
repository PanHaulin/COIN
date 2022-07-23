from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch import optim
from torchvision.models import resnet50, resnet18
from encoders.ResNet import resnet56


# from torchinfo import summary
from s_dbw import S_Dbw
import time
from collections import Counter

class BaseModel(pl.LightningModule):
    def __init__(
        self,
        encoder,
        num_classes: int,
        max_epochs: int,
        train_batch: int,
        optimizer: str,
        lr: float,
        weight_decay: float,
        momentum: float,
        accumulate_grad_batches: int,
        scheduler: str,
        save_epochs: list = [],
        encoder_weights: str = None,
        **kwargs,
    ):
        """Base model that implements all basic operations for all self-supervised methods.
        It adds shared arguments, extract basic learnable parameters, creates optimizers
        and schedulers, implements basic training_step for any number of crops,
        trains the online classifier and implements validation_step.
        现所有自监督方法的所有基本操作的基本模型。
        它添加共享参数，提取基本的可学习参数，创建优化器和调度器，
        为任意数量的作物实施基本的训练步骤，训练在线分类器并实施验证步骤。
        """

        super().__init__()

        # training related
        self.num_classes = num_classes
        self.max_epochs = max_epochs
        self.batch_size = train_batch
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.accumulate_grad_batches = accumulate_grad_batches
        self.scheduler = scheduler
        # self.epochs = epochs

        self.encoder_name = encoder
        
        # feats and label to cal metrics
        self.feats = None
        self.labels = None

        # all the other parameters
        self.extra_args = kwargs


        # initialize encoder
        if encoder == 'resnet50':
            self.encoder = resnet50(num_classes=num_classes)
            if encoder_weights == None:
                encoder_weights = "/home/pc/utils/weights/ssl/pretrain_moco_v2.pkl"
            self.features_dim = self.encoder.inplanes
        elif encoder == 'resnet18':
            self.encoder = resnet18(num_classes=num_classes)
            # load pretrained params
            if encoder_weights == None:
                encoder_weights = "/home/pc/utils/weights/ssl/pretrain_byol_res18.pth"
            self.features_dim = self.encoder.inplanes
        elif encoder == 'resnet56':
            self.encoder = resnet56(num_classes=num_classes)
            if encoder_weights == None:
                encoder_weights = "/home/pc/utils/weights/ssl/pretrain_byol_resnet56.pth"
            self.features_dim = self.encoder.in_planes
        else:
            assert encoder=='resnet50' "want encoder in [resnet50,resnet18,resnet56], but got {}".format(encoder)

        # load encoder weights
        self.encoder.load_state_dict(torch.load(encoder_weights), strict=False)

        # structure clip
        if "resnet" in encoder:
            del self.encoder.fc
            self.encoder.fc=lambda x:x
      
        self.classifier_input_dim = self.features_dim
        self.projector_input_dim = self.features_dim


    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:

        parser = parent_parser.add_argument_group("base")

        # encoder args
        SUPPORTED_NETWORKS = [ "resnet50", "resnet18", "resnet56"]

        parser.add_argument("--encoder", choices=SUPPORTED_NETWORKS, type=str)
        parser.add_argument("--encoder-weights", type=str, default=None)

        # general train
        parser.add_argument("--train_batch", type=int, default=64) # MAE fine-tuning: batch size=128， accumulate=1
        parser.add_argument("--test_batch", type=int, default=100)

        parser.add_argument("--num_workers", type=int, default=8)

        #logger, 默认使用tensorboard
        
        # optimizer
        SUPPORTED_OPTIMIZERS = ["sgd"]

        parser.add_argument("--optimizer", choices=SUPPORTED_OPTIMIZERS, type=str, required=True)
        parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
        parser.add_argument("--lr", type=float, default=0.01)
        parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

        # scheduler
        SUPPORTED_SCHEDULERS = [
            "cosine",
        ]

        parser.add_argument("--scheduler", choices=SUPPORTED_SCHEDULERS, type=str)

        return parent_parser
        # return parser

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        return [
            {"name": "encoder", "params": self.encoder.parameters()},
        ]

    def configure_optimizers(self) -> Tuple[List, List]:
        print("lr:{}, momentum:{}, wd:{}".format(self.lr, self.momentum, self.weight_decay))
        optimizer = optim.SGD(
                self.learnable_params,
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=True
                )
        scheduler_dict = {
                "scheduler": optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs),
                "interval": "epoch"
                }

        return {'optimizer':optimizer, 'lr_scheduler':scheduler_dict}
        

    def forward(self, X: torch.Tensor, *args, **kwargs):

        feats = self.encoder(X)

        return feats
    
    def on_train_start(self) -> None:
        super().on_train_start()

    def save_feats_labels(self, feats, labels):
        if self.feats is None:
            self.feats = feats.detach().cpu().data.float()
            self.labels = labels.detach().cpu().data
        else:
            self.feats = torch.cat((self.feats, feats.cpu().data), axis=0)
            self.labels = torch.cat((self.labels, labels.cpu().data), axis=0)

    
    def on_train_epoch_start(self) -> None:
        # 清空
        self.feats = None
        self.labels = None

        assert self.feats is None, "train epoch start: self.feats 未被清空"
        self.epoch_start_time = time.time()
        return super().on_train_epoch_start()

    def on_train_epoch_end(self) -> None:
        # 计时
        self.log('epoch_training_time', time.time() - self.epoch_start_time)

        # 计算s-dbw，仅在测试中计算，减少开销。
        # self._cal_sdbw('train')

        return super().on_train_epoch_end()
    
    def on_validation_epoch_start(self) -> None:
        # 清空
        self.feats = None
        self.labels = None

        assert self.feats is None, "validation epoch start: self.feats 未被清空"
        return super().on_validation_epoch_start()
    
    def on_validation_epoch_end(self) -> None:
        # print(sorted(Counter(self.labels).items()))

        # 计算s-dbw，仅在测试中计算，减少开销。
        self._cal_sdbw('val')

        return super().on_validation_epoch_end()
    
    def on_test_epoch_start(self) -> None:
        # 清空
        self.feats = None
        self.labels = None

        assert self.feats is None, "test epoch start: self.feats 未被清空"
        return super().on_test_epoch_start()
    
    def on_test_epoch_end(self) -> None:
        # 计算s-dbw
        self._cal_sdbw('test')

        return super().on_test_epoch_end()

    def _cal_sdbw(self, stage):
        sdbw_score = S_Dbw(self.feats.numpy(),self.labels.numpy(),centers_id=None,method='Tong',alg_noise='bind',centr='mean',nearest_centr=True,metric='euclidean')
        self.log(f'{stage}_sdbw_score', sdbw_score)
   
            