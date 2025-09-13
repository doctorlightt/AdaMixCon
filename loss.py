import itertools
import tabnanny
import torch
import torch.utils.data as Torchdata
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm
from scipy import io
from tools import *
import os
import time
import random
from model import VSSM
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):

        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = torch.tensor(alpha) if alpha is not None else None

    def forward(self, inputs, targets):

        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)

        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]) 
        ce_loss = -targets_one_hot * log_probs 
        
        loss = ce_loss

        if self.alpha is not None:
            alpha_factor = self.alpha.to(inputs.device)[targets]
            ce_loss = alpha_factor * ce_loss

        focal_weight = (1 - probs) ** self.gamma 
        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
        
def cal_loss(focal_loss, result, labels):
    result_fuse, result_1, result_2, result_3 = result
    loss_fuse = focal_loss(result_fuse, labels)
    loss_1, loss_2, loss_3 = 0, 0, 0
    if result_1 is None:
        loss_1 = 0
    else:
        for i in range(result_1.size(2)):
            loss_1 += focal_loss(result_1[:,:,i], labels)
    if result_2 is None:
        loss_2 = 0
    else:
        for i in range(result_2.size(2)):
            loss_2 += focal_loss(result_2[:,:,i], labels)
    if result_3 is None:
        loss_3 = 0
    else:
        for i in range(result_3.size(2)):
            loss_3 += focal_loss(result_3[:,:,i], labels)

    return loss_fuse + 0.5*(loss_1 + loss_2 + loss_3)