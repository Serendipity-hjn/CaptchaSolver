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

import string

# 导入自定义的类和函数
from CaptchaDataset import CaptchaDataset
from Model import Model
from Utils import train
from Utils import valid

# 定义字符集：包括数字、字母以及一个特殊字符'-'
characters = '-' + string.digits + string.ascii_uppercase
width, height, n_len, n_classes = 192, 64, 4, len(characters)  # 图像宽度，高度，验证码字符数，字符集大小
n_input_length = 12  # 输入特征的长度（图像的宽度）

# 打印配置信息
print(characters, width, height, n_len, n_classes)

# 创建一个示例数据集，生成一个验证码样本用于查看
dataset = CaptchaDataset(characters, 1, width, height, n_input_length, n_len)
image, target, input_length, label_length = dataset[0]  # 获取一个样本
print(''.join([characters[x] for x in target]), input_length, label_length)  # 打印目标字符与长度
to_pil_image(image)  # 将图像转换为PIL格式，方便查看

# 设置批量大小并创建训练集和验证集
batch_size = 70
train_set = CaptchaDataset(characters, 1000 * batch_size, width, height, n_input_length, n_len)
valid_set = CaptchaDataset(characters, 100 * batch_size, width, height, n_input_length, n_len)

# 使用DataLoader来加载训练和验证数据集
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=2)
valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=2)

# 创建模型并进行推理测试
model = Model(n_classes, input_shape=(3, height, width))
inputs = torch.zeros((32, 3, height, width))  # 输入一个假数据（32个批次的图像）
outputs = model(inputs)  # 得到模型的输出
outputs.shape  # 输出张量的形状

# 将模型移动到GPU上
model = Model(n_classes, input_shape=(3, height, width))
model = model.cuda()  # 将模型转移到GPU
model  # 输出模型信息

# 解码函数：将字符索引序列转换回验证码字符串
def decode(sequence):
    a = ''.join([characters[x] for x in sequence])  # 将索引转换为字符
    s = ''.join([x for j, x in enumerate(a[:-1]) if x != characters[0] and x != a[j+1]])  # 删除重复字符
    if len(s) == 0:
        return ''
    if a[-1] != characters[0] and s[-1] != a[-1]:  # 判断最后一个字符是否需要添加
        s += a[-1]
    return s

# 解码目标标签
def decode_target(sequence):
    return ''.join([characters[x] for x in sequence]).replace(' ', '')  # 还原字符并去除空格

# 计算准确率：通过比较目标标签和模型输出的解码结果
def calc_acc(target, output):
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)  # 获取输出的最大值索引
    target = target.cpu().numpy()  # 将目标标签转到CPU并转为numpy格式
    output_argmax = output_argmax.cpu().numpy()  # 将输出转到CPU并转为numpy格式
    # 比较目标和输出的解码结果是否一致
    a = np.array([decode_target(true) == decode(pred) for true, pred in zip(target, output_argmax)])
    return a.mean()  # 返回准确率

# 主程序：训练和验证过程
if __name__ == '__main__':
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), 1e-3, amsgrad=True)
    epochs = 6  # 设置训练的轮数
    # 训练和验证阶段（6轮）
    for epoch in range(1, epochs + 1):
        train(model, optimizer, epoch, train_loader, characters)  # 训练
        valid(model, optimizer, epoch, valid_loader, characters)  # 验证

    # 修改学习率并继续训练（3轮）
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, amsgrad=True)
    epochs = 3
    for epoch in range(1, epochs + 1):
        train(model, optimizer, epoch, train_loader, characters)  # 训练
        valid(model, optimizer, epoch, valid_loader, characters)  # 验证

    # 评估模式：使用模型进行推理并输出结果
    model.eval()  # 设置模型为评估模式
    output = model(image.unsqueeze(0).cuda())  # 输入图像进行推理
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)  # 获取解码后的输出
    do = True  # 控制循环
    while do or decode_target(target) == decode(output_argmax[0]):
        do = False  # 只执行一次
        image, target, input_length, label_length = dataset[0]  # 获取新的测试样本
        print('true:', decode_target(target))  # 打印真实标签

        output = model(image.unsqueeze(0).cuda())  # 再次进行推理
        output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)  # 获取解码后的输出
        print('pred:', decode(output_argmax[0]))  # 打印预测结果
    to_pil_image(image)  # 将图像转换为PIL格式并显示

    # 保存模型的状态字典
    torch.save(model.state_dict(), 'models/ctc3.pth')
