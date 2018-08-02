import os
import time
import copy
import torch
import torchvision
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image


def model_parser(model, fixed_weight=False):
    base_model = None
    if model == 'Googlenet':
        base_model = models.inception_v3(pretrained=True)
        network = GoogleNet(base_model, fixed_weight)
    elif model == 'Resnet':
        base_model = models.resnet34(pretrained=True)
        network = ResNet(base_model, fixed_weight)
    else:
        assert 'Unvalid Model'

    return network


class ResNet(nn.Module):
    def __init__(self, base_model, fixed_weight=False):
        super(ResNet, self).__init__()
        self.base_model = nn.Sequential(*list(base_model.children())[:-1])

        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.fc_last = nn.Linear(512, 2048, bias=True)
        self.fc_position = nn.Linear(2048, 3, bias=True)
        self.fc_rotation = nn.Linear(2048, 4, bias=True)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_last(x)
        position = self.fc_position(x)
        rotation = self.fc_rotation(x)

        return position, rotation


class GoogleNet(nn.Module):
    """ PoseNet using Inception V3 """
    def __init__(self, base_model, fixed_weight=False):
        super(GoogleNet, self).__init__()
        model = []

        model.append(base_model.Conv2d_1a_3x3)
        model.append(base_model.Conv2d_2a_3x3)
        model.append(base_model.Conv2d_2b_3x3)
        model.append(nn.MaxPool2d(kernel_size=3, stride=2))
        model.append(base_model.Conv2d_3b_1x1)
        model.append(base_model.Conv2d_4a_3x3)
        model.append(nn.MaxPool2d(kernel_size=3, stride=2))
        model.append(base_model.Mixed_5b)
        model.append(base_model.Mixed_5c)
        model.append(base_model.Mixed_5d)
        model.append(base_model.Mixed_6a)
        model.append(base_model.Mixed_6b)
        model.append(base_model.Mixed_6c)
        model.append(base_model.Mixed_6d)
        model.append(base_model.Mixed_6e)
        model.append(base_model.Mixed_7a)
        model.append(base_model.Mixed_7b)
        model.append(base_model.Mixed_7c)
        self.base_model = nn.Sequential(*model)

        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Out 2
        self.pos2 = nn.Linear(2048, 3, bias=True)
        self.ori2 = nn.Linear(2048, 4, bias=True)

    def forward(self, x):
        # 299 x 299 x 3
        x = self.base_model(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        pos = self.pos2(x)
        ori = self.ori2(x)

        return pos, ori
