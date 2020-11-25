import torch
import torchvision
from torch.utils.data import Dataset, TensorDataset, random_split, DataLoader
import numpy as np

# own experiment
class MyDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

init_dataset = TensorDataset(
    torch.randn(100, 24, 24),
    torch.randint(1, (100,1))
)

print(torch.randn(100, 24).shape)
print(torch.randint(2, (100,1)).shape)

lengths = [int(len(init_dataset)*0.8), int(len(init_dataset)*0.2)]
subsetA, subsetB = random_split(init_dataset, lengths)
datasetA = MyDataset(subsetA)
datasetB = MyDataset(subsetB)

dataloaderA = DataLoader(dataset=datasetA, batch_size=1, shuffle=True)
dataloaderB = DataLoader(dataset=datasetB, batch_size=1, shuffle=True)

dataiter = iter(dataloaderA)
features, labels = dataiter.next()
print(type(features))
print(features.shape)
print(labels.shape)

print(sum(1 for _ in dataloaderA))
print(sum(1 for _ in dataloaderB))

print("THEEND")