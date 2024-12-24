import sys
import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from captcha.image import ImageCaptcha
import random
import string
import torch
from torchvision.transforms.functional import to_tensor

# 添加 src 文件夹到 sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 从 src.Model 导入 Model 类
from Model import Model  # 确保从 src 文件夹加载 Model 类
from src.Utils import decode, decode_target

# 初始化 Flask 应用
app = Flask(__name__)

# 定义模型输入形状和类别数
characters = '-' + string.digits + string.ascii_uppercase
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型
model = Model(n_classes=len(characters), input_shape=(3, 64, 192))

# 加载模型的 state_dict
model.load_state_dict(torch.load('models/ctc3.pth', map_location=device))  # 加载权重
model.eval().to(device)  # 切换到评估模式并移动到设备

# 随机生成验证码文本
def generate_captcha_text(length=4):
    return ''.join(random.choice(characters[1:]) for _ in range(length))

# 使用模型进行预测
def predict_captcha(image_tensor):
    output = model(image_tensor.unsqueeze(0).to(device))
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    predicted_text = decode(output_argmax[0].cpu().numpy())
    return predicted_text

# 首页展示验证码
@app.route('/')
def index():
    captcha_text = generate_captcha_text()
    image = ImageCaptcha(width=192, height=64)
    captcha_image = image.generate_image(captcha_text)
    
    # 保存验证码的文本，以便后续验证
    with open("current_captcha.txt", "w") as f:
        f.write(captcha_text)
    
    # 转换为 PIL 图像以供前端显示
    captcha_image.save('static/captcha.png')
    
    return render_template('index.html', captcha_image='static/captcha.png')

# 验证用户输入的验证码
@app.route('/verify', methods=['POST'])
def verify():
    user_input = request.form.get('user_input')
    
    # 读取当前验证码文本
    with open("current_captcha.txt", "r") as f:
        correct_captcha = f.read().strip()

    # 如果用户输入正确
    if user_input == correct_captcha:
        return jsonify({"status": "success", "message": "验证码正确!"})
    else:
        return jsonify({"status": "error", "message": "验证码错误!"})

# 切换验证码图片
@app.route('/reload', methods=['GET'])
def reload_captcha():
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
