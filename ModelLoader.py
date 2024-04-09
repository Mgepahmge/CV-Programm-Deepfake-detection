import torch
import torch.nn as nn
import torch.optim as optim
from FileLoader import imageTensorLoader
import random
from tqdm import tqdm
from networks.mesonet import Meso4, MesoInception4
import matplotlib.pyplot as plt
from functools import singledispatchmethod


class ModelLoader:
    """
    模型加载器，用于加载模型并保存模型。
    """
    def __init__(self, model=Meso4, model_path=None):
        """
        :param model: 选择采用模型(Meso4/MesoInception4)
        :param model_path: 读取模型的路径
        """
        self.model = model()
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

    def train(self, fake_path, real_path, batch_size, num_epochs, iteration_nums=1000, learning_rate=0.001,
              weight_decay=1e-5):
        """
        开始训练

        参数:
        :param fake_path: 换脸图片训练集地址
        :param real_path: 真实图片训练集地址
        :param batch_size: 每次训练输入图片的数量
        :param num_epochs: 训练批次
        :param iteration_nums: 每批次迭代次数
        :param learning_rate: 学习率，默认为1e-3
        :param weight_decay: L2正则化参数，默认为1e-5
        """
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for epoch in range(num_epochs):
            fake_loader = imageTensorLoader(fake_path, 1, batch_size)
            real_loader = imageTensorLoader(real_path, 0, batch_size)
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
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    outputs = self.model.forward(inputs)
                    outputs = outputs.view(-1)
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    progress_bar.set_postfix(total_loss=total_loss / (iteration + 1))
                    progress_bar.update(1)
        print("训练结束！")

    def test(self, fake_path: str, real_path: str, iteration_nums=1000, draw_chart=False):
        """
        :param fake_path: 换脸图片训练集地址
        :param real_path: 真实图片训练集地址
        :param iteration_nums: 图片个数
        :param draw_chart: 是否画图
        :return:
        """
        fake_loader = imageTensorLoader(fake_path, 1, 1)
        real_loader = imageTensorLoader(real_path, 1, 1)
        outputs = []
        labels = []
        with tqdm(total=iteration_nums, desc=f'测试中', position=0,
                  leave=True) as progress_bar:
            for iteration in range(iteration_nums):
                random_number = random.random()
                try:
                    if random_number > 0.5:
                        inputs, label = next(fake_loader)
                    else:
                        inputs, label = next(real_loader)
                except StopIteration:
                    print("迭代数设置过大，测试集耗尽！")
                    break
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                outputs.append(torch.squeeze(self.model.forward(inputs)).item())
                labels.append(label[0])
                progress_bar.update(1)
        result_list = [1 if abs(x - y) <= 0.5 else 0 for x, y in zip(outputs, labels)]
        if draw_chart:
            x = range(1, iteration_nums + 1)
            # 绘制结果图
            plt.figure(figsize=(10, 5))
            plt.plot(x, result_list, label='预测值', color='orange')
            plt.plot(x, labels, label='真实值', color='blue')
            plt.title('预测结果与真实结果对比图')
            plt.xlabel('输入值编号')
            plt.ylabel('预测值/真实值')
            plt.legend()
            plt.show()

        return sum(result_list) / len(result_list)


if __name__ == '__main__':
    trainer = ModelLoader(model=Meso4, model_path="models/model.pth")
    test_loader = imageTensorLoader(
        folder_path="dataset/real_vs_fake/real-vs-fake/test/real", label=0,
        step=100)
    image, label = next(test_loader)
    result = trainer.model.forward(image)
    result = torch.squeeze(result)
    re = []
    for i in result.tolist():
        re.append(1 if i <= 0.5 else 0)
    print(sum(re) / len(re))
