import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

dataset = torchvision.dataset.MNIST(root='./data', transform=torchvision.transforms.ToTensor())

class WineDataSet(Dataset):

    def __init__(self, transform=None):
        # data loading
        xy = np.loadtxt('pytorch_turtorials/data/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = (xy[:, 1:]) #all samples all features, but not the wine
        self.y = (xy[:, [0]]) #all samples but only the wine
        self.n_samples = xy.shape[0]
        
        self.transform = transform

    def __getitem__(self, index):
        # dataset[0]
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)
        return sample 

    def __len__(self):
        # len(dataset)
        return self.n_samples

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets

dataset = WineDataSet(transform=ToTensor())

first_data = dataset[0]
features, labels = first_data
print(type(features))