import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备的测试数据集
test_dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# 测试数据集中的第一张图片及target
# img, target = test_dataset[0]
# print(img.shape)
# print(target)
# print(test_dataset.classes[target])

# tensorboard 查看数据
writer = SummaryWriter("../logs")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs)
        # print(targets)
        # writer.add_images("test_loader", imgs, step)
        writer.add_images("epoch: {}".format(epoch), imgs, step)
        step = step + 1

writer.close()

