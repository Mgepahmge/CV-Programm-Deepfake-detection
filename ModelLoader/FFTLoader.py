import random

import torch
from torch import nn, optim
from tqdm import tqdm

from FileProcessor import imageTensorLoader, getPowerSpectrumTensor
from networks import *


class FFTLoader:
    def __init__(self, model=FFTHead, model_path=None):
        """
        :param model: 采用模型FFTHead
        :param model_path: 读取模型的路径
        """
        self.model = model(179, 1, 256)
        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model)
            self.model.cuda()
        if model_path is not None:
            state_dict = torch.load(model_path)
            # 修改键，给每个键添加 'module.' 前缀
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = 'module.' + k
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)

    def save_model(self, save_path):
        """
        将当前模型的状态保存到指定的路径。

        参数:
        :param save_path: 模型应该被保存的文件路径。
        """
        if isinstance(self.model, nn.DataParallel):
            torch.save(self.model.module.state_dict(), save_path)
        else:
            torch.save(self.model.state_dict(), save_path)

    def train(self, fake_path, real_path,  num_epochs, iteration_nums=1000, learning_rate=0.001,
              weight_decay=1e-5):
        """
        开始训练

        参数:
        :param fake_path: 换脸图片训练集地址
        :param real_path: 真实图片训练集地址
        :param num_epochs: 训练批次
        :param iteration_nums: 每批次迭代次数
        :param learning_rate: 学习率，默认为1e-3
        :param weight_decay: L2正则化参数，默认为1e-5
        """
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for epoch in range(num_epochs):
            fake_loader = imageTensorLoader(fake_path, 1, 1)
            real_loader = imageTensorLoader(real_path, 0, 1)
            total_loss = 0
            with tqdm(total=iteration_nums, desc=f'批次 {epoch + 1}/{num_epochs}', position=0,
                      leave=True) as progress_bar:
                for iteration in range(iteration_nums):
                    random_number = random.random()
                    try:
                        if random_number > 0.5:
                            inputs, labels = next(fake_loader)
                        else:
                            inputs, labels = next(real_loader)
                    except StopIteration:
                        print('数据集已用尽，训练终止！')
                        break
                    labels = torch.tensor(labels, dtype=torch.float32)
                    inputs = getPowerSpectrumTensor(inputs)
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    outputs = self.model.forward(inputs)
                    outputs = outputs.squeeze()
                    labels = labels.squeeze()
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    progress_bar.set_postfix(total_loss=total_loss / (iteration + 1))
                    progress_bar.update(1)
        print("训练结束！")
