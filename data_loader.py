import os
import torch
import torchvision
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, image_path, metadata_path, mode):
        self.image_path = image_path
        self.metadata_path = metadata_path
        self.mode = mode
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


if __name__ == '__main__':
    image_path = '/mnt/data2/image_based_localization/posenet/KingsCollege'
    metadata_path = '/mnt/data2/image_based_localization/posenet/KingsCollege/dataset_train.txt'
    dataset = CustomDataset(image_path, metadata_path, 'train')



