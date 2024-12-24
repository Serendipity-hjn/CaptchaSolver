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

# 定义验证码数据集类
class CaptchaDataset(Dataset):
    def __init__(self, characters, length, width, height, input_length, label_length):
        """
        初始化验证码数据集类
        :param characters: 字符集，包含可用的字符
        :param length: 数据集的长度，即生成多少张图片
        :param width: 每张图片的宽度
        :param height: 每张图片的高度
        :param input_length: 输入序列的长度
        :param label_length: 标签序列的长度
        """
        super(CaptchaDataset, self).__init__()
        self.characters = characters  
        self.length = length  
        self.width = width  
        self.height = height  
        self.input_length = input_length  
        self.label_length = label_length  
        self.n_class = len(characters)  
        self.generator = ImageCaptcha(width=width, height=height)  

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        """
        获取数据集中的一个样本，包括图像和标签
        :param index: 索引，表示数据集中样本的位置
        :return: image, target, input_length, target_length
            - image: 生成的验证码图像（Tensor格式）
            - target: 对应的标签（Tensor格式，包含字符索引）
            - input_length: 输入序列的长度
            - target_length: 标签序列的长度
        """
        # 随机选择一个字符序列（验证码）
        random_str = ''.join([random.choice(self.characters[1:]) for j in range(self.label_length)])
        
        # 使用生成器生成对应验证码的图片
        image = to_tensor(self.generator.generate_image(random_str))
        
        # 将验证码的字符转换为字符索引，并转为Tensor
        target = torch.tensor([self.characters.find(x) for x in random_str], dtype=torch.long)
        
        # 创建输入序列长度的Tensor，通常为固定值
        input_length = torch.full(size=(1, ), fill_value=self.input_length, dtype=torch.long)
        
        # 创建标签长度的Tensor，通常等于验证码的字符数
        target_length = torch.full(size=(1, ), fill_value=self.label_length, dtype=torch.long)
        
        return image, target, input_length, target_length
