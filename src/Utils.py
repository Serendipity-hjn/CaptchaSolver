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


# 解码函数：将网络输出的索引序列转为字符序列
def decode(sequence, characters):
    """
    将输出的索引序列转换为字符序列，同时去除多余的字符（例如重复的字符）。
    :param sequence: 网络输出的索引序列
    :param characters: 字符集
    :return: 经过处理的字符序列
    """
    a = ''.join([characters[x] for x in sequence])  # 将索引转为字符
    s = ''.join([x for j, x in enumerate(a[:-1]) if x != characters[0] and x != a[j+1]])  # 去除多余字符
    if len(s) == 0:
        return ''  # 如果没有有效字符，则返回空字符串
    if a[-1] != characters[0] and s[-1] != a[-1]:
        s += a[-1]  # 如果最后一个字符不是填充字符，保留
    return s

# 解码目标函数：将目标索引序列转换为字符序列并去除空格
def decode_target(sequence, characters):
    """
    将目标的索引序列转换为字符序列，并去除空格字符。
    :param sequence: 目标索引序列
    :param characters: 字符集
    :return: 转换后的字符序列
    """
    return ''.join([characters[x] for x in sequence]).replace(' ', '')

# 计算准确率
def calc_acc(target, output, characters):
    """
    计算模型预测结果与真实标签之间的准确率。
    :param target: 真实标签
    :param output: 模型输出
    :param characters: 字符集
    :return: 准确率
    """
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)  # 对模型输出进行argmax，得到预测结果
    target = target.cpu().numpy()  # 将目标数据移至CPU并转换为numpy数组
    output_argmax = output_argmax.cpu().numpy()  # 将输出数据移至CPU并转换为numpy数组
    # 逐一比较预测结果与真实标签，计算正确率
    a = np.array([decode_target(true, characters) == decode(pred, characters) for true, pred in zip(target, output_argmax)])
    return a.mean()  # 返回准确率的均值


# 训练函数
def train(model, optimizer, epoch, dataloader, characters):
    """
    模型训练函数。
    :param model: 训练的模型
    :param optimizer: 优化器
    :param epoch: 当前的训练周期
    :param dataloader: 数据加载器
    :param characters: 字符集
    """
    model.train()  # 切换到训练模式
    loss_mean = 0  # 平均损失初始化
    acc_mean = 0  # 平均准确率初始化
    with tqdm(dataloader) as pbar:  # 使用tqdm显示进度条
        for batch_index, (data, target, input_lengths, target_lengths) in enumerate(pbar):
            data, target = data.cuda(), target.cuda()  # 将数据移到GPU
            
            optimizer.zero_grad()  # 清空梯度
            output = model(data)  # 前向传播，得到模型输出
            
            # 计算CTC损失
            output_log_softmax = F.log_softmax(output, dim=-1)
            loss = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)
            
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数

            loss = loss.item()  # 获取损失的数值
            acc = calc_acc(target, output, characters)  # 计算当前的准确率
            
            # 更新损失和准确率的滑动平均
            if batch_index == 0:
                loss_mean = loss
                acc_mean = acc
            
            loss_mean = 0.1 * loss + 0.9 * loss_mean  # 使用加权平均更新损失
            acc_mean = 0.1 * acc + 0.9 * acc_mean  # 使用加权平均更新准确率
            
            # 更新进度条的描述
            pbar.set_description(f'Epoch: {epoch} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')

# 验证函数
def valid(model, optimizer, epoch, dataloader, characters):
    """
    模型验证函数。
    :param model: 被验证的模型
    :param optimizer: 优化器（在验证过程中其实不需要用到）
    :param epoch: 当前的训练周期
    :param dataloader: 数据加载器
    :param characters: 字符集
    """
    model.eval()  # 切换到评估模式
    with tqdm(dataloader) as pbar, torch.no_grad():  # 不计算梯度
        loss_sum = 0  # 累积损失初始化
        acc_sum = 0  # 累积准确率初始化
        for batch_index, (data, target, input_lengths, target_lengths) in enumerate(pbar):
            data, target = data.cuda(), target.cuda()  # 将数据移到GPU
            
            output = model(data)  # 前向传播，得到模型输出
            output_log_softmax = F.log_softmax(output, dim=-1)  # 计算log-softmax
            loss = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)  # 计算CTC损失
            
            loss = loss.item()  # 获取损失的数值
            acc = calc_acc(target, output, characters)  # 计算准确率
            
            # 累加损失和准确率
            loss_sum += loss
            acc_sum += acc
            
            # 计算当前的平均损失和平均准确率
            loss_mean = loss_sum / (batch_index + 1)
            acc_mean = acc_sum / (batch_index + 1)
            
            # 更新进度条的描述
            pbar.set_description(f'Test : {epoch} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')
