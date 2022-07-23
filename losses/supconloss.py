from inspect import Parameter
import torch
from sklearn import preprocessing
import numpy as np

# def focal_SupConLoss(features, labels=None, mask=None, mixup=False, num_classes=10, temperature=0.07):
#     """
#     Partial codes are based on the implementation of supervised contrastive loss. 
#     import from https https://github.com/HobbitLong/SupContrast.
#     """
#     device = (torch.device('cuda')
#               if features.is_cuda
#               else torch.device('cpu'))

#     temperature = torch.clamp(temperature,0.07,1).type_as(features) if isinstance(temperature, torch.nn.Parameter) else temperature
    
#     base_temperature = temperature
#     batch_size = features.shape[0]
#     if labels is not None and mask is not None:
#         raise ValueError('Cannot define both `labels` and `mask`')
#     elif labels is None and mask is None:
#         mask = torch.eye(batch_size, dtype=torch.float32).to(device)
#         # mask = torch.eye(batch_size).type_as(features)
#     elif labels is not None:
#         if mixup:
#             if labels.size(1)>1:
#                 weight_index = 10**np.arange(num_classes)  
#                 weight_index = torch.tensor(weight_index).unsqueeze(1).to("cuda")
#                 # weight_index = torch.tensor(weight_index).unsqueeze(1).type_as(features)
#                 labels_ = labels.mm(weight_index.float()).squeeze(1)
#                 labels_ = labels_.detach().cpu().numpy()
#                 le = preprocessing.LabelEncoder()
#                 le.fit(labels_)
#                 labels = le.transform(labels_)
#                 labels=torch.unsqueeze(torch.tensor(labels),1)
#         labels = labels.contiguous().view(-1, 1) 
#         if labels.shape[0] != batch_size:
#             raise ValueError('Num of labels does not match num of features')
#         mask = torch.eq(labels, labels.T).float().to(device)
#         # mask = torch.eq(labels, labels.T).float()
#     else:
#         mask = mask.float().to(device)
#         # mask = mask.float()
   
#     anchor_feature = features.float()
#     contrast_feature = features.float()
#     anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),temperature)  
#     logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#     logits = anchor_dot_contrast - logits_max.detach()  
#     logits_mask = torch.scatter(
#         torch.ones_like(mask),  
#         1,
#         torch.arange(batch_size).view(-1, 1).to(device),
#         0
#     )
#     # logits_mask = torch.scatter(
#     #     torch.ones_like(mask),  
#     #     1,
#     #     torch.arange(batch_size).view(-1, 1),
#     #     0
#     # )
#     # mask = mask * logits_mask   

#     # compute log_prob
#     exp_logits = torch.exp(logits) * logits_mask  
#     log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) 
    
#     # compute weight
#     # weight = 1-torch.exp(log_prob)
#     weight = 1
    
#     # compute mean of log-likelihood over positive
#     mean_log_prob_pos = (weight * mask * log_prob).mean(1)
#     # !TODO 下面的形式才是原本的的实现形式
#     # mean_log_prob_pos = (weight * mask * log_prob).sum(1) / mask.sum(1)

#     # loss
#     mean_log_prob_pos = - (temperature / base_temperature) * mean_log_prob_pos
#     mean_log_prob_pos = mean_log_prob_pos.view(batch_size)
    
#     N_nonzeor = torch.nonzero(mask.sum(1)).shape[0]
#     loss = mean_log_prob_pos.sum()/N_nonzeor
#     if torch.isnan(loss):
#         print("nan contrastive loss")
#         loss=torch.zeros(1).to(device)   
#         # loss=torch.zeros(1)      
#     return loss

# deepseed and half
def focal_SupConLoss(features, labels=None, mask=None, mixup=False, num_classes=10, temperature=0.07):
    """
    Partial codes are based on the implementation of supervised contrastive loss. 
    import from https https://github.com/HobbitLong/SupContrast.
    """

    temperature = torch.clamp(temperature,0.07,1).type_as(features) if isinstance(temperature, torch.nn.Parameter) else temperature
    
    base_temperature = temperature
    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size).type_as(features)
    elif labels is not None:
        if mixup:
            if labels.size(1)>1:
                weight_index = 10**np.arange(num_classes)  
                weight_index = torch.tensor(weight_index).unsqueeze(1).type_as(features)
                labels_ = labels.mm(weight_index.type_as(features)).squeeze(1)
                labels_ = labels_.detach().cpu().numpy()
                le = preprocessing.LabelEncoder()
                le.fit(labels_)
                labels = le.transform(labels_)
                labels=torch.unsqueeze(torch.tensor(labels),1)
        labels = labels.contiguous().view(-1, 1) 
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).type_as(features)
    else:
        mask = mask.type_as(features)
   
    anchor_feature = features.type_as(features)
    contrast_feature = features.type_as(features)
    anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),temperature)  
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()  
    logits_mask = torch.scatter(
        torch.ones_like(mask),  
        1,
        torch.arange(batch_size).view(-1, 1).type_as(features).long(),
        0
    )

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask  
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) 
    
    # compute weight
    # weight = 1-torch.exp(log_prob)
    weight = 1
    
    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (weight * mask * log_prob).mean(1)
    # !TODO 下面的形式才是原本的的实现形式
    # mean_log_prob_pos = (weight * mask * log_prob).sum(1) / mask.sum(1)

    # loss
    mean_log_prob_pos = - (temperature / base_temperature) * mean_log_prob_pos
    mean_log_prob_pos = mean_log_prob_pos.view(batch_size)
    
    N_nonzeor = torch.nonzero(mask.sum(1)).shape[0]
    loss = mean_log_prob_pos.sum()/N_nonzeor
    if torch.isnan(loss):
        print("nan contrastive loss")
        loss=torch.zeros(1)
    return loss.type_as(features)