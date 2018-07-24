import os
import torch
import torchvision
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, datasets
import torch.nn.functional as F
from torchvision.models import inception
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, image_path, metadata_path, mode, transform):
        self.image_path = image_path
        self.metadata_path = metadata_path
        self.mode = mode
        self.transform = transform
        raw_lines = open(self.metadata_path, 'r').readlines()
        self.lines = raw_lines[3:]

        print(self.lines.__len__())
        print(self.lines[0])

        self.test_filenames = []
        self.test_poses = []
        self.train_filenames = []
        self.train_poses = []

        for i, line in enumerate(self.lines):
            splits = line.split()
            filename = splits[0]
            values = splits[1:]
            values = list(map(lambda x: float(x.replace(",", "")), values))

            filename = os.path.join(self.image_path, filename)

            if self.mode == 'train':
                if i < 100:
                    self.test_filenames.append(filename)
                    self.test_poses.append(values)
                else:
                    self.train_filenames.append(filename)
                    self.train_poses.append(values)
            elif self.mode == 'test':
                self.test_filenames.append(filename)
                self.test_poses.append(values)
            else:
                assert 'Unavailable mode'

        self.num_train = self.train_filenames.__len__()
        self.num_test = self.test_filenames.__len__()
        print("Number of Train", self.num_train)
        print("Number of Test", self.num_test)

    def __getitem__(self, index):
        if self.mode == 'train':
            image = Image.open(self.train_filenames[index])
            pose = self.train_poses[index]
        elif self.mode == 'test':
            image = Image.open(self.test_filenames[index])
            pose = self.test_poses[index]

        return self.transform(image), pose

    def __len__(self):
        if self.mode == 'train':
            num_data = self.num_train
        elif self.mode == 'test':
            num_data = self.num_test
        return num_data


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = nn.BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = nn.BasicConv2d(128, 768, kernel_size=5)
        self.fc = nn.Linear(768, 1024)


    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = F.dropout(x, training=self.training)
        x = self.fc(x)
        # 1000
        return x

class PoseNet(nn.Module):
    """ PoseNet using Inception V3 """
    def __init__(self, InceptionV3):
        super(PoseNet, self).__init__()
        self.base_model = InceptionV3

        self.Conv2d_1a_3x3 = InceptionV3.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = InceptionV3.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = InceptionV3.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = InceptionV3.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = InceptionV3.Conv2d_4a_3x3
        self.Mixed_5b = InceptionV3.Mixed_5b
        self.Mixed_5c = InceptionV3.Mixed_5c
        self.Mixed_5d = InceptionV3.Mixed_5d
        self.Mixed_6a = InceptionV3.Mixed_6a
        self.Mixed_6b = InceptionV3.Mixed_6b
        self.Mixed_6c = InceptionV3.Mixed_6c
        self.Mixed_6d = InceptionV3.Mixed_6d
        self.Mixed_6e = InceptionV3.Mixed_6e
        self.Mixed_7a = InceptionV3.Mixed_7a
        self.Mixed_7b = InceptionV3.Mixed_7b
        self.Mixed_7c = InceptionV3.Mixed_7c

        # Out 1
        self.conv0 = nn.BasicConv2d(768, 128, kernel_size=1)
        self.conv1 = nn.BasicConv2d(128, 768, kernel_size=5)
        self.fc = nn.Linear(768, 1024)
        self.pos1 = nn.Linear(2048, 3, bias=False)
        self.ori1 = nn.Linear(2048, 4, bias=False)

        # Out 2
        self.pos2 = nn.Linear(2048, 3, bias=False)
        self.ori2 = nn.Linear(2048, 4, bias=False)


    def forward(self, x):
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        aux = self.AuxLogits(x)


        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)


class PoseNetSimple(nn.Module):
    """ Simple PoseNet using Inception V3 """
    def __init__(self, InceptionV3):
        super(PoseNetSimple, self).__init__()
        self.model = nn.Sequential(*list(InceptionV3.children())[:-1])
        self.model.aux_logits = False
        self.pos = nn.Linear(2048, 3, bias=False)
        self.ori = nn.Linear(2048, 4, bias=False)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, kernel_size=8)
        x = F.dropout(x, training=self.training)
        x = x.view(x.size(0), -1)
        pos = self.pos(x)
        ori = self.ori(x)

        return pos, ori


if __name__ == '__main__':
    image_path = '/mnt/data2/image_based_localization/posenet/KingsCollege'
    metadata_path = '/mnt/data2/image_based_localization/posenet/KingsCollege/dataset_train.txt'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = CustomDataset(image_path, metadata_path, 'train', transform)

    print(dataset.__len__())

    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = models.inception_v3(pretrained=True)
    model.aux_logits = False
    posenet = PoseNetSimple(model)

    model2 = models.alexnet(pretrained=True)

    dummy = torch.randn(2, 3, 299, 299)

    model_mine = nn.Sequential(*list(model.children())[:-1])
    print(posenet(dummy))




