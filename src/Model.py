import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image

from captcha.image import ImageCaptcha
from tqdm import tqdm
import random
import numpy as np
from collections import OrderedDict

class Model(nn.Module):
    def __init__(self, n_classes, input_shape=(3, 64, 128)):
        """
        初始化模型，包括卷积层、LSTM层和全连接层。
        :param n_classes: 输出类别数（验证码字符集的大小）
        :param input_shape: 输入图像的形状，默认为(3, 64, 128)，即3个通道，高64，宽128
        """
        super(Model, self).__init__()
        self.input_shape = input_shape  # 输入的形状
        channels = [32, 64, 128, 256, 256]  # 每个卷积块的输出通道数
        layers = [2, 2, 2, 2, 2]  # 每个卷积块的卷积层数
        kernels = [3, 3, 3, 3, 3]  # 卷积核大小（这里都是3x3）
        pools = [2, 2, 2, 2, (2, 1)]  # 池化层的大小
        
        # 定义卷积块、批归一化、ReLU激活函数等
        modules = OrderedDict()

        # 定义卷积块的构建方式：卷积层 -> 批归一化 -> ReLU
        def cba(name, in_channels, out_channels, kernel_size):
            modules[f'conv{name}'] = nn.Conv2d(in_channels, out_channels, kernel_size,
                                                padding=(1, 1) if kernel_size == 3 else 0)
            modules[f'bn{name}'] = nn.BatchNorm2d(out_channels)  # 批归一化
            modules[f'relu{name}'] = nn.ReLU(inplace=True)  # ReLU激活函数

        last_channel = 3  # 输入图像的通道数为3（RGB）
        # 构建多个卷积块
        for block, (n_channel, n_layer, n_kernel, k_pool) in enumerate(zip(channels, layers, kernels, pools)):
            for layer in range(1, n_layer + 1):
                # 每个卷积块包含多个卷积层
                cba(f'{block+1}{layer}', last_channel, n_channel, n_kernel)
                last_channel = n_channel
            # 每个卷积块后面跟一个池化层
            modules[f'pool{block + 1}'] = nn.MaxPool2d(k_pool)
        
        # 添加Dropout层，防止过拟合
        modules[f'dropout'] = nn.Dropout(0.25, inplace=True)
        
        # 将所有的卷积层、池化层、Dropout等组合成一个顺序容器
        self.cnn = nn.Sequential(modules)
        
        # 定义LSTM层，用于处理卷积后提取的特征
        self.lstm = nn.LSTM(input_size=self.infer_features(), hidden_size=128, num_layers=2, bidirectional=True)
        
        # 定义全连接层，用于将LSTM的输出映射到字符集大小（n_classes）
        self.fc = nn.Linear(in_features=256, out_features=n_classes)

    def infer_features(self):
        """
        推断通过卷积网络处理后，得到的特征数量。
        :return: 特征的数量，用于LSTM输入的维度
        """
        x = torch.zeros((1,)+self.input_shape)  # 创建一个虚拟输入图像（大小为1的batch）
        x = self.cnn(x)  # 通过卷积网络处理
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # 展平卷积后的特征图
        return x.shape[1]  # 返回特征数量，用于LSTM的输入大小

    def forward(self, x):
        """
        前向传播函数：从输入图像到输出预测。
        :param x: 输入的图像（Batch, Channel, Height, Width）
        :return: 模型的输出预测
        """
        x = self.cnn(x)  # 通过卷积层提取特征
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # 展平特征图，准备输入到LSTM
        x = x.permute(2, 0, 1)  # LSTM的输入需要是（序列长度，batch_size，特征数）
        x, _ = self.lstm(x)  # LSTM处理特征
        x = self.fc(x)  # 通过全连接层映射到输出类别
        return x
