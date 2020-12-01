import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
#from plotcm import plot_confusion_matrix

import pdb


torch.set_printoptions(linewidth=120)

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=240)
        self.fc2 = nn.Linear(in_features=240, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=120)
        self.fc4 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # (1) input layer
        t = t
        
        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (4) hidden linear layer
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) hidden linear layer
        t = self.fc3(t)
        t = F.relu(t)

        # (7) hidden linear layer
        t = self.fc4(t)
        t = F.relu(t)

        # (8) output layer
        t = self.out(t)
        #t = F.softmax(t, dim=1)

        return t


train_set = torchvision.datasets.FashionMNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

network = Network()


train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=True)
data= next(iter(train_loader))
mean = data[0].mean()
std = data[0].std()
mean, std

train_set = torchvision.datasets.FashionMNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
        , transforms.Normalize(mean,std)
        #, transforms.Normalize((mean,),(std,))
    ])
)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)

optimizer = optim.Adam(network.parameters(), lr=0.001)

for epoch in range(50):

    total_loss = 0
    total_correct = 0

    for batch in train_loader: # Get Batch
        
        images, labels = batch
        
        images = (images - mean) / std
        
        preds = network(images) # Pass Batch
        loss = F.cross_entropy(preds, labels) # Calculate Loss

        optimizer.zero_grad()
        loss.backward() # Calculate Gradients
        optimizer.step() # Update Weights

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)

    print(
        "epoch", epoch, 
        "total_correct:", total_correct, 
        "loss:", total_loss
    )


train_set = torchvision.datasets.FashionMNIST(
    root='./data'
    ,train=False
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)


train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=True)

for batch in train_loader: # Get Batch

        preds = network(images) # Pass Batch

        optimizer.zero_grad()
       
        total_correct += get_num_correct(preds, labels)

print("total_correct:", total_correct)   