
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
from typing import Tuple
from PIL import Image
import re
import matplotlib.pyplot as plt
import cv2
import numpy as np

class My_Dataset(Dataset):
    def __init__(self, transform=None, train=True):

        if train:
            train = "train"
        else:
            train = "test"

        self.edge_image_ids = []
        edge_folder_list = ["beautiful landscape edge", "landscape edge", "korean_landscape edge", "landscape photo edge", "kaggle edge", "natural landscape edge", "twilight landscape edge"]
        for i in edge_folder_list:
            self.edge_image_ids += glob.glob("images/"+train+"/"+i + "/*")
#        self.edge_image_ids  =self.edge_image_ids[:10]

        self.color_image_ids = []
        color_folder_list = ["beautiful landscape", "landscape", "korean_landscape", "landscape photo", "kaggle", "natural landscape", "twilight landscape"]
        for i in color_folder_list:
            self.color_image_ids += glob.glob("images/"+train+"/"+i + "/*")
#        self.color_image_ids  =self.color_image_ids[:10]

        self.transforms = transform

    def __len__(self):
        return len(self.edge_image_ids)

    def __getitem__(self, index: int) -> Tuple[Tensor]:

        edge_image_id = self.edge_image_ids[index]
        edge_image = Image.open(edge_image_id)

        color_image_id = self.color_image_ids[index]
        color_image = Image.open(color_image_id).convert("RGB")

        if self.transforms:
            edge_image = self.transforms(edge_image)
            color_image = self.transforms(color_image)

        return edge_image, color_image


# train_transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.Resize((324, 324)),
#     transforms.ToTensor()
# ])
#
# test_transform = transforms.Compose([
#     transforms.Resize((324, 324)),
#     transforms.ToTensor()
# ])
# train_set = My_Dataset(train_transform, train=True)
# test_set = My_Dataset(test_transform, train=False)
#
# training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=1, shuffle=True)
# testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1,
#                                  shuffle=False)
#
# for i, (edge, color) in enumerate(testing_data_loader):
#     plt.imshow(color.squeeze(0).permute(1,2,0))
#     plt.show()
#
# plt.imshow(x.permute(1,2,0))
# plt.show()
# plt.imshow(y.permute(1,2,0))
# plt.show()
