from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import cv2

# python的用法 --> tensor数据类型
# 通过 transforms.ToTensor 去看两个问题
# 1. transforms 该如何使用（python）
# 2. 为什么我们需要Tensor数据类型

img_path = "../dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)
# print(img)

writer = SummaryWriter("../logs/transforms")

# 1. transforms 该如何使用（python）
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
# print(tensor_img)

writer.add_image("Tensor_img", tensor_img)
writer.close()

