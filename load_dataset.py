import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10, FashionMNIST
from torch.utils.data import DataLoader, Dataset

class OneClassFashionMNIST(Dataset):
    def __init__(self, root=".", download=True, transform=torchvision.transforms.ToTensor(), category=0):
        self.data = FashionMNIST(root=root, download=download, transform=transform)
        self.category = category
        self.indices = [i for i, (_, label) in enumerate(self.data) if label == self.category]

    def __getitem__(self, index):
        image, _ = self.data[self.indices[index]]
        return image

    def __len__(self):
        return len(self.indices)


class OneClassCIFAR10(Dataset):
    def __init__(self, root=".", download=True, transform=torchvision.transforms.ToTensor(), category='airplane'):
        self.data = CIFAR10(root=root, download=download)
        self.transform = transform
        self.category = category
        self.class_to_idx = self.data.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.indices = [i for i, (_, label) in enumerate(self.data) if self.idx_to_class[label] == self.category]

    def __getitem__(self, index):
        image, _ = self.data[self.indices[index]]
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.indices)


def show_images(dataloader, num_samples=12, cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15,15)) 
    for i, batch in enumerate(dataloader):
        if i == num_samples:
            break
        img = batch[0]
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.imshow(img.permute(1, 2, 0))

# dataset = OneClassCIFAR10(root=".", download=True, category='airplane')
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
# show_images(dataloader)