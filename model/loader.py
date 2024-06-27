import os
import numpy as np
import pandas as pd
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
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


# load printed_digit dataset
pdig_dataset = DigitDataset(annotations_file='./data/printed_digit/labels.csv',
                          img_dir='./data/printed_digit/',
                          transform=transform)

train_indices, test_indices = train_test_split(np.arange(len(pdig_dataset)), 
                                               test_size=0.2, random_state=42)

pdig_trainset = Subset(pdig_dataset, train_indices)
pdig_testset = Subset(pdig_dataset, test_indices)

pdig_trainloader = DataLoader(pdig_trainset, batch_size=10, shuffle=True)
pdig_testloader = DataLoader(pdig_testset, batch_size=10, shuffle=True)


# Load MNIST dataset
mnist_trainset = MNIST(root="./data", train=True, download=True, transform=transform)
mnist_testset = MNIST(root="./data", train=False, download=True, transform=transform)

mnist_trainloader = DataLoader(mnist_trainset, batch_size=10, shuffle=True)
mnist_testloader = DataLoader(mnist_testset, batch_size=10, shuffle=False)


# Load hoda dataset
hoda_dataset = DigitDataset(annotations_file='./data/hoda/labels.csv',
                            img_dir='./data/hoda/',
                            transform=transform)

hoda_train_idx, hoda_test_idx = train_test_split(np.arange(len(hoda_dataset)),
                                                  test_size=0.2, random_state=42)

hoda_trainset = Subset(hoda_dataset, hoda_train_idx)
hoda_testset = Subset(hoda_dataset, hoda_test_idx)

hoda_trainloader = DataLoader(hoda_trainset, batch_size=10, shuffle=True)
hoda_testloader = DataLoader(hoda_testset, batch_size=10, shuffle=True)



# Load farsi_printed_digit dataset
fpdig_dataset = DigitDataset(annotations_file='./data/farsi_printed_digit/labels.csv',
                            img_dir='./data/farsi_printed_digit/',
                            transform=transform)

fpdig_train_idx, fpdig_test_idx = train_test_split(np.arange(len(fpdig_dataset)),
                                                  test_size=0.2, random_state=42)

fpdig_trainset = Subset(fpdig_dataset, fpdig_train_idx)
fpdig_testset = Subset(fpdig_dataset, fpdig_test_idx)

fpdig_trainloader = DataLoader(fpdig_trainset, batch_size=10, shuffle=True)
fpdig_testloader = DataLoader(fpdig_testset, batch_size=10, shuffle=True)



data_loader = {
    "printed_digit": {"train": pdig_trainloader, "test": pdig_testloader},
    "mnist": {"train": mnist_trainloader, "test": mnist_testloader},
    "hoda": {"train": hoda_trainloader, "test": hoda_testloader},
    "farsi_printed_digit" : {"train": fpdig_trainloader, "test": fpdig_testloader}
}
