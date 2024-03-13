import torch
import torch.nn as nn
import torch.optim as optim
from ImageLoader import imageLoader
from MainModel import MainModel
import random
from tqdm import tqdm


class Trainer:
    def __init__(self, fake_path, real_path, batch_size, model_path=None):
        self.fake_loader = imageLoader(fake_path, 1, batch_size)
        self.real_loader = imageLoader(real_path, 0, batch_size)
        self.model = MainModel()
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
            with tqdm(total=iteration_nums, desc=f'批次 {epoch + 1}/{num_epochs}', position=0,
                      leave=True) as progress_bar:
                for iteration in range(iteration_nums):
                    random_number = random.random()
                    try:
                        if random_number > 0.5:
                            inputs, labels = next(self.fake_loader)
                        else:
                            inputs, labels = next(self.real_loader)
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
                    progress_bar.set_postfix(loss=loss.item())
                    progress_bar.update(1)
        print("训练结束！")


if __name__ == '__main__':
    fake = "D:/Python项目/CV-Programm-Deepfake-detection/dataset/real_vs_fake/real-vs-fake/test/fake"
    real = "D:/Python项目/CV-Programm-Deepfake-detection/dataset/real_vs_fake/real-vs-fake/test/real"
    trainer = Trainer(fake_path=fake, real_path=real, batch_size=1)
    trainer.train(num_epochs=10, iteration_nums=100)
