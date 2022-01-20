# p14 torchvision 中的数据集使用
import cv2
import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_daset = torchvision.datasets.CIFAR10(root="../dataset", train=True, download=True, transform=dataset_transform)
test_daset = torchvision.datasets.CIFAR10(root="../dataset", train=False, download=True, transform=dataset_transform)

# print(test_daset[0])
# print(test_daset.classes)
#
# img, target = test_daset[0]
# print(img)
# print(target)
# print(test_daset.classes[target])
# img.show() # PIL
#
# print(test_daset[0])

writer = SummaryWriter(log_dir="../logs")
for i in range(10):
    img, target = test_daset[i]
    # writer.add_image(tag="test_dataset", img_tensor=img, global_step=i)
    writer.add_image("test_dataset", img, i)

writer.close()

