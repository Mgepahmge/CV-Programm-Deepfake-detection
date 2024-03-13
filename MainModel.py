import torch
import torch.nn as nn
import torch.nn.functional as F
from FeatureExtractionModel import FeatureExtractionModel
from ClassfyModel import ClassificationMLP


class MainModel(nn.Module):
    def __init__(self):
        super(MainModel, self).__init__()
        self.feature_extractor = FeatureExtractionModel()
        self.classifier = ClassificationMLP()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = MainModel()
    x = torch.randn(1, 3, 256, 256)
    y = model.forward(x)
    torch.nn.Sequential()
    print(y)
