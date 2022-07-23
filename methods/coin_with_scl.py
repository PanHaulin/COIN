import argparse
from typing import Any, Dict, List
import torch
import torch.nn as nn
from methods.base import BaseModel
from torchmetrics.functional import accuracy
import torch.nn.functional as F
from losses.supconloss import focal_SupConLoss

class Contrastor(nn.Module): 
    def __init__(self, n_inputs, n_outputs):
        super(Contrastor, self).__init__()
        self.input = nn.Linear(n_inputs, n_inputs) 
        self.output = nn.Linear(n_inputs, n_outputs) 
    def forward(self, x):
        x = self.input(x) 
        x = F.relu(x) 
        x = self.output(x)
        return F.normalize(x)

class COIN(BaseModel):
    def __init__(self, proj_name, output_dim: int, eta_weight: int, temperature: float, alpha: float, **kwargs):
        super().__init__(**kwargs)
        # projector
        print('use projector: ', proj_name)
        self.projector = Contrastor(self.projector_input_dim, output_dim)

        # classifier
        self.classifier = nn.Linear(self.classifier_input_dim, self.num_classes, bias=False)

        self.eta_weight = eta_weight

        self.temperature = temperature
        
        self.stage1_epochs = alpha * 100

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser):
        parent_parser = super(COIN, COIN).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("contrast")

        # projector
        parser.add_argument("--proj_name", type=str, default="Contrastor")
        parser.add_argument("--output_dim", type=int, default=256)
        # params
        parser.add_argument('--eta_weight', default=0.1, type=float, help='trade-off parameter')
        parser.add_argument('--temperature', default=0.3, type=float)
        parser.add_argument('--alpha', default=0.7, type=float)
        return parent_parser
    
    def forward(self, X, *args, **kwargs):
        feats = super().forward(X, *args, **kwargs)
        return self.base_forward(feats, **kwargs)

    def base_forward(self, out):
        feats = out
        contrastor_features = self.projector(feats)
        prediction  = self.classifier(feats) 
        return feats, contrastor_features, prediction
    
    def training_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        inputs, targets = batch

        # inputs, targets = batch
        feats, contrast_features,  outputs  = self.forward(inputs)

        # 保存 encoder feats
        self.save_feats_labels(feats=feats, labels=targets)
        
        classification_loss = F.cross_entropy(outputs, targets.long())
        contrastive_loss = focal_SupConLoss(contrast_features, targets, num_classes=self.num_classes, temperature=self.temperature)

        if self.trainer.current_epoch < self.stage1_epochs:
            total_loss = contrastive_loss
        else:
            total_loss = classification_loss + self.eta_weight * contrastive_loss

        # preds = torch.argmax(logits, dim=1)
        stage = 'train'
        acc1 = accuracy(outputs.data, targets, top_k=1)
        acc5 = accuracy(outputs.data, targets, top_k=5)

        if stage is None:
            pass
        else:
            self.log(f'{stage}_classification_loss', classification_loss, prog_bar=True)
            self.log(f'{stage}_contrastive_loss', contrastive_loss, prog_bar=True)
            self.log(f'{stage}_total_loss', total_loss, prog_bar=True)
            self.log(f'{stage}_acc1', acc1, prog_bar=True)
            self.log(f'{stage}_acc5', acc5, prog_bar=False)
            
        return total_loss
    
    def _shared_step(self, inputs, targets, stage=None):
        
        # inputs, targets = batch
        feats, contrast_features,  outputs  = self.forward(inputs)

        # 保存 encoder feats
        self.save_feats_labels(feats=feats, labels=targets)
        
        classification_loss = F.cross_entropy(outputs, targets.long())
        contrastive_loss = focal_SupConLoss(contrast_features, targets, num_classes=self.num_classes, temperature=self.temperature)
        # total_loss = classification_loss + self.eta_weight * contrastive_loss

        if self.trainer.current_epoch < self.stage1_epochs:
            total_loss = classification_loss
        else:
            total_loss = classification_loss + self.eta_weight * contrastive_loss

        # preds = torch.argmax(logits, dim=1)
        acc1 = accuracy(outputs.data, targets, top_k=1)
        acc5 = accuracy(outputs.data, targets, top_k=5)

        if stage is None:
            pass
        else:
            self.log(f'{stage}_classification_loss', classification_loss)
            self.log(f'{stage}_contrastive_loss', contrastive_loss)
            self.log(f'{stage}_total_loss', total_loss, prog_bar=True)
            self.log(f'{stage}_acc1', acc1, prog_bar=True)
            self.log(f'{stage}_acc5', acc5, prog_bar=False)
            
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        self._shared_step(inputs, targets, 'val')
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        self._shared_step(inputs, targets, 'test')
    
    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """
        params = [
            {"name": "classifier", "params": self.classifier.parameters()},
            {"name": "projector", "params":self.projector.parameters()},
        ]

        return super().learnable_params + params
    
        




