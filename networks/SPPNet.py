import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SPPLayer(nn.Module):
    def __init__(self, in_channels, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()
        self.in_channels = in_channels
        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        spp_tensors = []

        for i in range(self.num_levels):
            level = i + 1
            kernel_size = (math.ceil(height / level), math.ceil(width / level))
            stride = (math.ceil(height / level), math.ceil(width / level))
            padding = (
                math.floor((kernel_size[0] * level - height + 1) / 2),
                math.floor((kernel_size[1] * level - width + 1) / 2))

            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)
            elif self.pool_type == 'avg_pool':
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)
            tensor = F.interpolate(tensor, size=(height, width), mode='bilinear', align_corners=True)

            spp_tensors.append(tensor)

        spp_tensors.append(x)
        spp_output = torch.cat(spp_tensors, dim=1)

        return spp_output


if __name__ == '__main__':
    spp_layer = SPPLayer(in_channels=2048, num_levels=4, pool_type='max_pool')
    x = torch.randn(1, 2048, 8, 8)
    print(spp_layer.forward(x).shape)
