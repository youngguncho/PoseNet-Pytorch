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


def model_parser(model, fixed_weight=False, dropout_rate=0.0):
    base_model = None

    if model == 'Googlenet':
        base_model = models.inception_v3(pretrained=True)
        network = GoogleNet(base_model, fixed_weight, dropout_rate)
    elif model == 'Resnet':
        base_model = models.resnet34(pretrained=True)
        # network = ResNet(base_model, fixed_weight, dropout_rate)
        network = PoseNet(base_model)
    elif model == 'ResnetSimple':
        base_model = models.resnet34(pretrained=True)
        network = ResNetSimple(base_model, fixed_weight)
    else:
        assert 'Unvalid Model'

    return network


class PoseNet(nn.Module):
    def __init__(self, original_model):
        super(PoseNet, self).__init__()

        # feature들을 마지막 fully connected layer를 제외화고 ResNet으로 부터 가져옴
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.regressor = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # nn.Linear(2048, 7)
        )
        self.trans_regressor = nn.Sequential(
            nn.Linear(2048, 3)
        )
        self.rotation_regressor = nn.Sequential(
            nn.Linear(2048, 4)
        )
        self.modelName = 'resnet'

        # Freeze those weights
        # for p in self.features.parameters():
        #     p.requires_grad = False

        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(0)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        for m in self.trans_regressor.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(0)
                m.weight.data[0].normal_(0, 0.5)
                m.weight.data[1].normal_(0, 0.5)
                m.weight.data[2].normal_(0, 0.1)
                m.bias.data.zero_()

        for m in self.rotation_regressor.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(0)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, inpt):
        f = self.features(inpt)
        f = f.view(f.size(0), -1)
        y = self.regressor(f)
        trans = self.trans_regressor(y)
        rotation = self.rotation_regressor(y)

        return trans, rotation


class ResNet(nn.Module):
    def __init__(self, base_model, fixed_weight=False, dropout_rate=0.0):
        super(ResNet, self).__init__()
        self.dropout_rate = dropout_rate
        feat_in = base_model.fc.in_features

        self.base_model = nn.Sequential(*list(base_model.children())[:-1])
        # self.base_model = base_model

        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.fc_last = nn.Linear(feat_in, 2048, bias=True)
        self.fc_position = nn.Linear(2048, 3, bias=True)
        self.fc_rotation = nn.Linear(2048, 4, bias=True)

        init_modules = [self.fc_last, self.fc_position, self.fc_rotation]

        # for module in init_modules:
        #     if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        #         nn.init.kaiming_normal(module.weight.data)
        #         if module.bias is not None:
        #             nn.init.constant(module.bias.data, 0)

        nn.init.normal_(self.fc_last.weight.data, 0, 0.01)
        nn.init.constant_(self.fc_last.bias.data, 0)

        nn.init.normal_(self.fc_position.weight.data, 0, 0.5)
        nn.init.constant_(self.fc_position.bias.data, 0)

        nn.init.normal_(self.fc_rotation.weight.data, 0, 0.01)
        nn.init.constant_(self.fc_rotation.bias.data, 0)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_last(x)
        x = F.relu(x)

        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        position = self.fc_position(x)
        rotation = self.fc_rotation(x)

        return position, rotation


class ResNetSimple(nn.Module):
    def __init__(self, base_model, fixed_weight=False, dropout_rate=0.0):
        super(ResNetSimple, self).__init__()
        self.dropout_rate = dropout_rate
        feat_in = base_model.fc.in_features

        self.base_model = nn.Sequential(*list(base_model.children())[:-1])
        # self.base_model = base_model

        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # self.fc_last = nn.Linear(feat_in, 2048, bias=True)
        self.fc_position = nn.Linear(feat_in, 3, bias=False)
        self.fc_rotation = nn.Linear(feat_in, 4, bias=False)

        init_modules = [self.fc_position, self.fc_rotation]

        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight.data)
                if module.bias is not None:
                    nn.init.constant(module.bias.data, 0)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        position = self.fc_position(x)
        rotation = self.fc_rotation(x)

        return position, rotation

class GoogleNet(nn.Module):
    """ PoseNet using Inception V3 """
    def __init__(self, base_model, fixed_weight=False, dropout_rate = 0.0):
        super(GoogleNet, self).__init__()
        self.dropout_rate =dropout_rate

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
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        pos = self.pos2(x)
        ori = self.ori2(x)

        return pos, ori
