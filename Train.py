import torch
import torch.nn as nn
import torch.optim as optim
from ImageLoader import imageLoader
import random
from tqdm import tqdm
from networks.mesonet import Meso4, MesoInception4
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, fake_path, real_path, batch_size, model=Meso4, model_path=None):
        """
        :param fake_path: 换脸图片数据集路径
        :param real_path: 真实图片数据集路径
        :param batch_size: 每次训练的批次数
        :param model: 选择采用模型(Meso4/MesoInception4)
        :param model_path: 读取模型的路径
        """
        self.fake_path = fake_path
        self.real_path = real_path
        self.batch_size = batch_size
        self.model = model()
        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model)
            self.model.cuda()
        if model_path is not None:
            self.model = torch.load(model_path)

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

    def train(self, num_epochs, iteration_nums=1000, learning_rate=0.001, weight_decay=1e-5):
        """
        开始训练

        参数:
        :param num_epochs: 训练批次
        :param iteration_nums: 每批次迭代次数
        :param learning_rate: 学习率，默认为1e-3
        :param weight_decay: L2正则化参数，默认为1e-5
        """
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for epoch in range(num_epochs):
            fake_loader = imageLoader(self.fake_path, 1, self.batch_size)
            real_loader = imageLoader(self.real_path, 0, self.batch_size)
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
        fake_loader = imageLoader(fake_path, 1, 1)
        real_loader = imageLoader(real_path, 1, 1)
        criterion = nn.BCELoss()
        loss_list = []
        result_list = []
        for iteration in range(iteration_nums):
            random_number = random.random()
            try:
                if random_number > 0.5:
                    inputs, labels = next(fake_loader)
                else:
                    inputs, labels = next(real_loader)
            except StopIteration:
                print("迭代数设置过大，测试集耗尽！")
                break
            labels = torch.tensor(labels, dtype=torch.float32)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = self.model.forward(inputs)
            outputs = outputs.view(-1)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item)
            outputs = torch.squeeze(outputs)
            labels = torch.squeeze(labels)
            result_list.append(1 if abs(labels.item() - outputs.item()) > 0.5 else 0)
        total_loss = sum(loss_list) / len(loss_list)
        arc = sum(result_list) / len(result_list)
        if draw_chart:
            x = list(range(1, iteration_nums + 1))
            plt.plot(x, loss_list)
            plt.plot(x, result_list)
            plt.show()
        return total_loss, arc


if __name__ == '__main__':
    fake = "D:/Python项目/CV-Programm-Deepfake-detection/dataset/real_vs_fake/real-vs-fake/train/fake"
    real = "D:/Python项目/CV-Programm-Deepfake-detection/dataset/real_vs_fake/real-vs-fake/train/real"
    trainer = Trainer(fake_path=fake, real_path=real, batch_size=1)
    trainer.train(num_epochs=10, iteration_nums=50000)
    trainer.save_model("D:/Python项目/CV-Programm-Deepfake-detection/model.pth")
