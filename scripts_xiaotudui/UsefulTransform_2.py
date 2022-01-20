import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("../logs/transforms")
img = Image.open("../dataset/train/ants_image/0013035.jpg")
print(img)

# ToTensor的使用
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize
# output[channel] = (input[channel] - mean[channel]) / std[channel]
print(img_tensor[0][0][0])
# trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# trans_norm = transforms.Normalize([1, 3, 5], [3, 2, 1])
trans_norm = transforms.Normalize([3, 6, 8], [2, 5, 8])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm, 2)

writer.close()

