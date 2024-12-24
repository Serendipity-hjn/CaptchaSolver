import torch
from torchvision.transforms.functional import to_pil_image
from CaptchaDataset import CaptchaDataset
from Model import Model
from main import characters, width, height, n_input_length, n_len, decode, decode_target

# 加载模型
model_path = 'models/ctc3.pth'
model = torch.load(model_path)
model.eval()  

dataset = CaptchaDataset(characters, 1, width, height, n_input_length, n_len)

image, target, input_length, label_length = dataset[0]
image = image.unsqueeze(0).cuda()  # 添加批次维度并移动到 GPU

with torch.no_grad():
    output = model(image)
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    predicted_text = decode(output_argmax[0].cpu().numpy())


true_text = decode_target(target.cpu().numpy())
print(f"True: {true_text}")
print(f"Predicted: {predicted_text}")


to_pil_image(image.squeeze(0).cpu()).show()
