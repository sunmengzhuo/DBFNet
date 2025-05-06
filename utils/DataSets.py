import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import pandas as pd
import numpy as np
import config
from models import DualBranchResNet

# 定义数据集类
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, folder1, folder2, folder3, label_path, transform=None):
        self.folder1 = folder1
        self.folder2 = folder2
        self.folder3 = folder3
        self.label_path = label_path
        self.transform = transform
        self.data1 = sorted(os.listdir(folder1))
        self.data2 = sorted(os.listdir(folder2))
        self.data3 = sorted(os.listdir(folder3))
        self.label_data = pd.read_excel(label_path)

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, index):
        img_1_path = os.path.join(self.folder1, self.data1[index])
        img_2_path = os.path.join(self.folder2, self.data2[index])
        img_3_path = os.path.join(self.folder3, self.data3[index])

        # 读取图像并转换为灰度图
        img_1 = Image.open(img_1_path).convert('L')
        img_2 = Image.open(img_2_path).convert('L')
        img_3 = Image.open(img_3_path).convert('L')

        # 如果需要，将灰度图像转换为3通道（RGB）
        img_1 = img_1.convert("RGB")
        img_2 = img_2.convert("RGB")
        img_3 = img_3.convert("RGB")

        # 读取标签
        case_num = self.data1[index][:-4]  # 假设文件名后四个字符是扩展名
        label = torch.tensor(self.label_data.loc[self.label_data['病案号'] == int(case_num), 'MVI'].values.item())



        # 应用图像预处理（在转换为Tensor之前）
        if self.transform:
            # img_1, img_2, img_3 = self.transform(img_1, img_2, img_3)
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
            img_3 = self.transform(img_3)
        else:
            # 如果没有定义transform，直接将PIL图像转换为Tensor
            img_1 = transforms.ToTensor()(img_1)
            img_2 = transforms.ToTensor()(img_2)
            img_3 = transforms.ToTensor()(img_3)
        img_1 = torch.cat((img_1, img_3), dim=0)
        img_2 = torch.cat((img_2, img_3), dim=0)
        return img_1, img_2, label
