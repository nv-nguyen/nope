import torch.nn as nn
import torch
import pytorch_lightning as pl
import logging
import torch.nn.functional as F
from src.utils.visualization_utils import put_image_to_grid
from torchvision.utils import make_grid, save_image
import os
import wandb
import torchvision.transforms as transforms
from src.model.loss import GeodesicError
import multiprocessing
from src.poses.vsd import vsd_obj
from functools import partial
import time
from tqdm import tqdm
import numpy as np


def conv1x1(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False
    )


class InfoNCE(nn.Module):
    def __init__(self, tau=0.1, extra_contrast_type=None):
        super(InfoNCE, self).__init__()
        self.tau = tau
        self.extra_contrast_type = extra_contrast_type

    def forward(self, pos_sim, neg_sim, sim_extra_obj=None):
        """
        neg_sim: BxB
        pos_sim: Bx1
        sim_extra: BxB use extra object as negative
        """
        b = neg_sim.shape[0]
        logits = (1 - torch.eye(b)).type_as(neg_sim) * neg_sim + torch.eye(b).type_as(
            pos_sim
        ) * pos_sim

        labels = torch.arange(b, dtype=torch.long, device=logits.device)
        if sim_extra_obj is not None:
            sim_extra_obj = sim_extra_obj[:b]
            if self.extra_contrast_type == "BOP_ShapeNet":
                # Add more negative samples by taking pairs (BOP, ShapeNet)
                logits = torch.cat((logits, sim_extra_obj), dim=1)
            elif self.extra_contrast_type == "ShapeNet_ShapeNet":
                # Add more negative samples by taking pairs (ShapeNet, ShapeNet), duplicate the positive samples from BOP to get Identity matrix
                extra_logits = (1 - torch.eye(b)).type_as(
                    sim_extra_obj
                ) * sim_extra_obj + torch.eye(b).type_as(pos_sim) * pos_sim
                logits = torch.cat((logits, extra_logits), dim=0)  # 2BxB
                extra_labels = torch.arange(
                    b, dtype=torch.long, device=logits.device
                ).cuda()
                labels = torch.cat(
                    (labels, extra_labels), dim=0
                )  # 2B as [Identity, Identity]
        logits = logits / self.tau
        loss = F.cross_entropy(logits, labels)
        return [torch.mean(pos_sim), torch.mean(neg_sim), loss]


class OcclusionAwareSimilarity(nn.Module):
    def __init__(self, threshold):
        super(OcclusionAwareSimilarity, self).__init__()
        self.threshold = threshold

    def forward(self, similarity_matrix):
        indicator_zero = similarity_matrix <= self.threshold
        similarity_matrix[indicator_zero] = 0
        return similarity_matrix


class BaseFeatureExtractor(pl.LightningModule):
    def __init__(self, descriptor_size, threshold, **kwargs):

        # define the network
        super(BaseFeatureExtractor, self).__init__()
        self.loss = InfoNCE()
        self.metric = GeodesicError()
        self.occlusion_sim = OcclusionAwareSimilarity(threshold=threshold)
        self.sim_distance = nn.CosineSimilarity(dim=1)  # eps=1e-2

        # remove all the pooling layers, fc layers with conv1x1
        layer1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=(8, 8), stride=(2, 2)
        )
        layer2 = nn.Conv2d(in_channels=16, out_channels=7, kernel_size=(5, 5))
        projector = nn.Sequential(
            conv1x1(7, 256), nn.ReLU(), conv1x1(256, descriptor_size)
        )
        self.encoder = nn.Sequential(layer1, nn.ReLU(), layer2, nn.ReLU(), projector)


    def forward(self, x):
        feat = self.backbone(x)
        return feat