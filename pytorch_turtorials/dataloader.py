import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# Terminology
''' 
epoch = 1 forward and backward over ALL samples
batch size = number of training samples in one forward and backward pass
number of iterations = number of passes, each pass using [batchsize] number of samples
e.g. 100 samples, batch =20, --> 100/5 5 iterations of for 1 epoch

Update over all samples takes a long time, therefore dataLoader is used to divide in batche
And only update over a batch
'''

class WineDataSet(Dataset):

    def __init__(self):
        # data loading
        xy = np.loadtxt('pytorch_turtorials/data/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:]) #all samples all features, but not the wine
        self.y = torch.from_numpy(xy[:, [0]]) #all samples but only the wine
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples

dataset = WineDataSet()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True) #still errors with num_workers uses multiple processors 

# Training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)

print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward, backward, update
        if (i+1) % 5 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')
