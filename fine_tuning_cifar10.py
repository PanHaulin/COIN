from pathlib import Path
import os
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import argparse
import torchvision.datasets as datasets
from utils.datamodule import DataModule
from utils.checkpointers import MonitorCheckpointer
import time
from torchvision.transforms import transforms
from methods.coin_with_scl import COIN
# sys.path.append('.')

def get_transform(is_train: bool, normalization: transforms.Normalize = None) -> transforms.Compose:
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224), # interpolation=PIL.Image.BICUBIC
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(224 + 32), # interpolation=PIL.Image.BICUBIC
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])

def train(args):
    # set status
    args.status = 'fine-tuning'

    # fix the seed for reproducibility
    if args.manualSeed is None:
        args.manualSeed = 42

    seed_everything(args.manualSeed)

    # 定义transform
    print('==> Preparing dataset %s' % args.dataset) 
    transform_train = get_transform(is_train=True)
    transform_test = get_transform(is_train=False)
        
    

    # 设置数据集及类别数
    dataset_method = datasets.CIFAR10
    num_classes = 10

    # train and val datamodule
    dm = DataModule(
        DATASET_METHOD=dataset_method,
        data_dir=args.data_dir,
        train_batch=args.train_batch,
        test_batch=args.test_batch,
        num_workers=args.num_workers,
        train_transform=transform_train, 
        val_transform=transform_test,
        test_transform=transform_test,
        )
    args.num_classes= num_classes

    # resume
    # use Trainer.args: resume_from_checkpoint

    # initialize logger
    print("==> initializing logger TensorBoard")
    logger = TensorBoardLogger("tb_logs", name="my_model")
    
    # callbacks
    print("==> initializing callbacks")
    callbacks = []
    # 设置存储位置
    if args.resume:
        assert args.resume_dir != None, 'must set resume_dir when resume from checkpoint'
        args.logdir = args.resume_dir
    else:
        args.logdir = f'{args.checkpoints_savedir}/{args.method}/{time.strftime("%y%m%d%H%M%S")}/'
    
    print("logger_dir: ", args.logdir)
    args.checkpoint_save_last = True # training 时必然存最后一轮权重，并在训练结束后删除
    ckpt = MonitorCheckpointer.get_instance(args)
    callbacks.append(ckpt)
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)
    
    # log args
    logger.log_hyperparams(args)
    
    # initialize Trainer
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        callbacks=callbacks, 
        num_sanity_val_steps=0,
        gpus=1,
        auto_select_gpus=True,
    )

    # create model 
    model = COIN(**args.__dict__)
    
    print("==> creating model COIN")
    print("==> using method class '{}'".format(COIN))
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    
    # fit
    print("==> training and validating")
    if args.resume:
        if args.resume_checkpoint_filename != None:
            ckpt_path = Path(args.logdir) / args.resume_checkpoint_filename
        else:
            ckpt_path = Path(args.logdir) / 'last.ckpt'
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
    else:
        trainer.fit(model,datamodule=dm)

    # 删除last权重
    if not args.checkpoint_save_last:
        os.remove(Path(args.logdir) / 'last.ckpt')

    print("==> loading and testing best model")
    
    # 读取最好模型
    if not args.fast_dev_run:
        best_model_path = ckpt.best_model_path
        print("==> loading best_path: ",best_model_path)
        model = COIN.load_from_checkpoint(checkpoint_path=best_model_path, **args.__dict__)

    # testing
    trainer.test(model, datamodule=dm)


def parse_finetuning_args():
    parser = argparse.ArgumentParser()

    # dataset params
    SUPPORTED_DATASETS = [
        "cifar10",
    ]
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="cifar10", type=str, required=False)
    parser.add_argument("--num_classes", type=int, default=10, required=False)
    parser.add_argument("--data_dir", type=str, default="/home/pc/utils/dataset/cifar-10-batches-py", required=False)

    # add pytorch lightning trainer args
    parser = Trainer.add_argparse_args(parser)

    # add method-specific arguments
    parser.add_argument("--method", type=str, default='COIN', required=False)

    # add model specific args
    parser = COIN.add_model_specific_args(parser)
    
    # add checkpointer args
    parser = MonitorCheckpointer.add_checkpointer_args(parser)

    # others finetuning args
    parser.add_argument('--manualSeed', type=int, help='manual seed')


    args = parser.parse_args()
    return args

def main():
    args = parse_finetuning_args()
    train(args)

if __name__ == '__main__':
    main()
