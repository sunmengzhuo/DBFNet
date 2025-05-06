import torch
import torch.nn as nn
from models import resnet1
from models import resnet2
from models import resnet3


class LiverSE_Channel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bn = nn.BatchNorm2d(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=3, padding=1),  # 添加压缩比
            nn.ReLU(),
            nn.Conv2d(dim // 16, dim, kernel_size=3, padding=1),  # 恢复通道数
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.mlp(self.bn(x))


class LiverSE_Spatial(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.att(x)  # 自动广播到原通道数


class DualBranchConvNext(nn.Module):
    def __init__(self):
        super(DualBranchConvNext, self).__init__()
        self.resnet1 = resnet1.resnet18()
        self.resnet2 = resnet2.resnet18()
        self.spatialAtt = LiverSE_Spatial(512)
        self.channelAtt1 = LiverSE_Channel(512)
        self.channelAtt2 = LiverSE_Channel(512)
        # self.resnet3 = resnet3.resnet18()
        # 融合层
        self.fc1 = nn.Linear(1536, 256)  # 假设两个分支的输出特征维度为512
        self.fc2 = nn.Linear(256, 2)  # 假设两个分支的输出特征维度为512
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x1, x2):
        f1, g1 = self.resnet1(x1)  # Branch1的输出特征
        f2, g2 = self.resnet2(x2)  # Branch2的输出特征
        # f3, g3 = self.resnet2(x3)  # Branch2的输出特征
        loss1 = g1
        loss2 = g2
        combined_features = g1 + g2  # 在通道维度上相加
        # g1 = torch.flatten(g1, 1)

        # g2 = torch.flatten(g2, 1)
        g1 = self.channelAtt1(g1)
        g1 = self.global_avg_pool(g1)
        g1 = torch.flatten(g1, 1)

        g2 = self.channelAtt2(g2)
        g2 = self.global_avg_pool(g2)
        g2 = torch.flatten(g2, 1)
        # f3 = self.branch2(x3)  # Branch2的输出特征
        # 特征融合
        # g3 = self.channelAtt1(g3)
        # g3 = self.global_avg_pool(g3)
        # g3 = torch.flatten(g3, 1)

        x_fused = self.spatialAtt(combined_features)
        x_fused = self.global_avg_pool(x_fused)
        x_fused = torch.flatten(x_fused, 1)

        out = torch.cat((g1, g2, x_fused), dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out, loss1, loss2


# 用于测试模型
if __name__ == "__main__":
    model = DualBranchConvNext()
    x1 = torch.randn(8, 3, 224, 224)  # 第一个分支输入
    x2 = torch.randn(8, 3, 224, 224)  # 第二个分支输入
    output = model(x1, x2)
    print(output.shape)  # 输出应该是 [8, 2]
