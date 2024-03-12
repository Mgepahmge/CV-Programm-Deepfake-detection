import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.Xception import Xception
from networks.SENet import SENModule
from networks.SPPNet import SPPLayer


class FeatureExtractionModel(nn.Module):
    def __init__(self, num_levels_spp=4):
        super(FeatureExtractionModel, self).__init__()
        self.xception = Xception()
        self.sen_module = SENModule(in_channels=728, out_channels=728)
        self.spp_layer = SPPLayer(in_channels=2048, num_levels=num_levels_spp, pool_type='max_pool')
        self.conv_reduce = nn.Conv2d(2048 + (2048 * (num_levels_spp + 1)) + 728, 512, kernel_size=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化层

    def forward(self, x):
        feature_b, feature_a = self.xception(x)
        feature_c = self.sen_module(feature_a)
        feature_d = self.spp_layer(feature_b)
        feature_e = F.interpolate(feature_c, size=(8, 8), mode='bilinear', align_corners=True)
        concatenated_features = torch.cat([feature_b, feature_d, feature_e], dim=1)
        reduced_features = self.conv_reduce(concatenated_features)
        pooled_features = self.global_avg_pool(reduced_features)
        output_feature_vector = pooled_features.view(pooled_features.size(0), -1)  # Flatten the features
        return output_feature_vector


# Example usage
if __name__ == '__main__':
    model = FeatureExtractionModel()
    input_image = torch.randn(1, 3, 256, 256)
    output_feature_vector = model(input_image)
    print(output_feature_vector.shape)
    print(output_feature_vector)
