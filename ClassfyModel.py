import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationMLP(nn.Module):
    def __init__(self, input_size=512):
        super(ClassificationMLP, self).__init__()
        # MLP特征提取部分
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # 分类模块
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 特征提取
        features = self.feature_extractor(x)
        # 分类
        output = self.classifier(features)
        return output
