from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("../logs")
image_path = "../dataset/train/ants_image/6240329_72c01e663e.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)


writer.add_image("test", img_array, 1, dataformats='HWC')
# writer.add_scalar()
# y = 2 * x
for i in range(100):
    writer.add_scalar("y=2x", 4*i**2, i)
writer.close()


