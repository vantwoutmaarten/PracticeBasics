import torch
import torch.nn as nn
import torchvision 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# Device config
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
hidden_size = 500
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

input_size = 28
sequence_length = 28
num_layers = 2 # This means stacking 2 RNN's

# Fully connected neural network with one hidden layer
train_dataset = torchvision.datasets.MNIST(root='./data',train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.MNIST(root='./data',train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Normal net
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # GRU
        # self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # GRU
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # not sure if this Linear layer is only for many-to-one-classification
        self.fc = nn.Linear(hidden_size, num_classes)

        # if batch_first=True, then x -> batch_size, sequence_length, input_length 

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        # cell state for lstm 

        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        # Normal net
        # out, _ = self.rnn(x, h0)  
        # GRU
        # out, _ = self.gru(x, h0)  
        # LSTM
        out, _ = self.lstm(x, (h0, c0))

        # out --> batch_size, seq__length, hidden_size
        # out (N,28,128)
        
        # Only the last timestep is needed
        # out (N, 128)
        out = out[:, -1, :]

        out = self.fc(out)
        return out

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train
n_total_steps = len(train_loader)
# loop over the dataset multiple times
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images,labels) in enumerate(train_loader, 0):
        images = images.reshape(-1, sequence_length, input_size)
        images, labels = images.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f'Loss: {running_loss}')

print('Finished Training')

# print('Finished Training')

# Test the model
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size)
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        # max returns (value, index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
    
    acc = 100.0 * n_correct/n_samples
    print(f'Tets Accuracy of the network on the 10000 test images: {acc} %')

