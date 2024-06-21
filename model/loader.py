import os
import numpy as np
import pandas as pd
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image




class DigitDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("L")  # Ensure the image is in Gray format
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.Resize((28, 28)), 
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,))
])


# load my dataset
my_dataset = DigitDataset(annotations_file='./data/printed_digit/labels.csv',
                       img_dir='./data/printed_digit/',
                       transform=transform)

train_indices, test_indices = train_test_split(np.arange(len(my_dataset)), test_size=0.2, random_state=42)

my_trainset = Subset(my_dataset, train_indices)
my_testset = Subset(my_dataset, test_indices)

my_trainloader = DataLoader(my_trainset, batch_size=10, shuffle=True)
my_testloader = DataLoader(my_testset, batch_size=10, shuffle=True)


# Load MNIST dataset
data_root = "./data"
mnist_trainset = MNIST(root=data_root, train=True, download=True, transform=transform)
mnist_testset = MNIST(root=data_root, train=False, download=True, transform=transform)

mnist_trainloader = DataLoader(mnist_trainset, batch_size=10, shuffle=True)
mnist_testloader = DataLoader(mnist_testset, batch_size=10, shuffle=False)


data_loader = {
    "my_dataset": {"train": my_trainloader, "test": my_testloader},
    "mnist": {"train": mnist_trainloader, "test": mnist_testloader},
}
