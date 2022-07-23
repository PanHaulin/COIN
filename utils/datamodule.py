from typing import Optional
import torchvision.datasets as datasets
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch

class DataModule(pl.LightningDataModule):
    def __init__(self, DATASET_METHOD, data_dir:str, train_batch:int, test_batch:int ,num_workers: int, train_transform, val_transform, test_transform, **kwargs):
        super().__init__()
        # self.save_hyperparameters()
        self.data_dir = data_dir
        # self.batch_size = batch_size
        self.train_batch = train_batch
        self.test_batch = test_batch
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.DATASET_METHOD = DATASET_METHOD
        self.kwargs = kwargs
    
    def prepare_data(self):
        # 下载数据
        self.DATASET_METHOD(root=self.data_dir, train=True, download=True, **self.kwargs)
        self.DATASET_METHOD(root=self.data_dir, train=False, download=True, **self.kwargs)
    
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.trainset = self.DATASET_METHOD(root=self.data_dir, train=True, transform=self.train_transform, **self.kwargs)
            self.validset = self.DATASET_METHOD(root=self.data_dir, train=False, transform=self.val_transform, **self.kwargs)
        if stage == "test" or stage is None:
            self.testset = self.DATASET_METHOD(root=self.data_dir, train=False, transform=self.test_transform, **self.kwargs)
    
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.train_batch, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.test_batch, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.test_batch, shuffle=False, num_workers=self.num_workers)
