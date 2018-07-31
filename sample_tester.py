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


if __name__ == '__main__':

    base_model = models.resnet34(pretrained=True)

    print(base_model)
    #
    # print(base_model.__sizeof__())

    model = nn.Sequential(*list(base_model.children())[:-1])
    model2 = list(base_model.modules())


    print('-' * 50)
    print('-' * 50)

    print(list(nn.Sequential(nn.Linear(10, 20), nn.ReLU()).modules()))
    # print((base_model.modules()))


    print('-' * 50)
    print('-' * 50)

    # print((base_model.children()))

