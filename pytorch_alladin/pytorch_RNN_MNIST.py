# Create a fully connected neural net in pytorch
# steps
## Import
## Create fully connected network
## Set Device 
## Hyperparameters
## Load Data
## Init network
## Loss and optimizer
## Train Network
## Check accuracy on training & test to see how good our model is

# Import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# model = NN(784, 10)
# x = torch.rand(64, 784).cuda() # 64 random examples with 28x28=784 features
# print(model(x).shape)  #input x in the model returns output by applying forward(rough idea)

# Set Device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #assuming gpu is available

# (Hyper-)parameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256

num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

# Create a RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True) # with batch_first true, send in # Nxtime_seqxfeatures
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)
  
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward Prop
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

# Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True) #dataset arrives as numpy, therefore transform and download if not present in folder yet
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) #after a epoch shuffles the data, so that images per batch are different
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True) 
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Init network
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Train Network
for epoch in range(num_epochs): #in one epoch the network sees al the images in the dataset
    for batch_idx, (data, targets) in enumerate(train_loader): #bring the data of the current batch to a device
        print(device)
        data = data.to(device=device).squeeze(1) #remove the 1 in Nx1x28x28
        targets = targets.to(device=device)

        print(data.shape) # torch.Size([64, 1, 28, 28])  64 the samples in a batch, 1 black or white color channel, 28x28 LxW of MNIST

        # forward 
        scores = model(data).to(device=device)
        loss = criterion(scores, targets)

        # backword prop
        optimizer.zero_grad() #Set all the gradients to zero for each batch so that the backprop of each batch is not stored. 
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

# Check accuracy on training & test to see how good our model is
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on train data")
    else: 
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()  

    with torch.no_grad(): # no gradients for backprop have to be calculated
        for x,y in loader: #not sure yet what is loader?
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x) #64x10
            _, predictions = scores.max(1) #index of the value
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    
    model.train() #when you evaluate the model during training then after eval(), train() should be used. 
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
