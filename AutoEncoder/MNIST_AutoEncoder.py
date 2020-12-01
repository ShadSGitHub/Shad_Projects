import numpy as np

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
#%matplotlib inline
import torch.nn as nn
import torch.nn.functional as F

batch_size = 100 #- 500
epochs = 20
learning_rate = 1e-3

# t shape is 1,12,4,4
# torch.randn(1,12,4,4) for the testing

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.enoch = Encoder()
        self.denoch = Decoder()

    def passToEnoch(self, image):

        bottleNeck = self.enoch(image)

        return bottleNeck

    def passToDenoch(self, BN):

        GenImg = self.denoch(BN)

        return GenImg





class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.up_convT1 = nn.ConvTranspose2d(in_channels=12, out_channels=12, kernel_size=2, stride=1)
        self.up_convT2 = nn.ConvTranspose2d(in_channels=12, out_channels=12, kernel_size=2, stride=1)
        self.up_convT3 = nn.ConvTranspose2d(in_channels=12, out_channels=12, kernel_size=2, stride=1)

        self.up_convT4 = nn.ConvTranspose2d(in_channels=12, out_channels=6, kernel_size=2, stride=2)
        self.up_convT5 = nn.ConvTranspose2d(in_channels=6, out_channels=1, kernel_size=2, stride=2)

    def forward(self, t):

        t = self.up_convT1(t)
        t = F.relu(t)
        t = self.up_convT2(t)
        t = F.relu(t)
        t = self.up_convT3(t)
        t = F.relu(t)

        t = self.up_convT4(t)
        t = F.relu(t)

        t = self.up_convT5(t)
        #t = F.relu(t)

        #try segmoid next
        t = torch.sigmoid(t)

        return t

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)


    def forward(self, t):
       
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        return t
'''
train_set = torchvision.datasets.CIFAR10(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)   



train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
for batch in train_loader: # Get Batch
    I,L = batch
    print(I.shape)
    break

'''

train_set = torchvision.datasets.MNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)

network = Model()

optimizer = optim.Adam(network.parameters(), lr=0.001)

criterion = nn.BCELoss()

for epoch in range(20):

    total_loss_1 = 0
    total_loss_2 = 0


    for batch in train_loader: # Get Batch
        
        images, _ = batch
        
        outputs = network.passToEnoch(images) # Pass Batch
        outputs = network.passToDenoch(outputs)
        loss = criterion(outputs, images) # Calculate Loss

        optimizer.zero_grad()
        loss.backward() # Calculate Gradients
        optimizer.step() # Update Weights

        total_loss_1 += loss.item()
        total_loss_2 += loss.item()*images.size(0)

    print(
        "epoch", epoch, 
        "total_loss_1:", total_loss_1, 
        "total_loss_2:", total_loss_2
    )

torch.save(network.state_dict(), 'AutoEncoder.pth')



test_set = torchvision.datasets.MNIST(
    root='./data'
    ,train=False
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=10, shuffle=True)

test_examples = None

with torch.no_grad():
    network.eval()
    for batch_features in test_loader:
        label = batch_features[1]
        batch_features = batch_features[0]

        outputs = network.passToEnoch(batch_features)
        reconstruction = network.passToDenoch(outputs)
        for x in range(10):
            ax = plt.subplot(2, 10, x + 1)
            plt.imshow(transforms.ToPILImage()(batch_features[x]))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax = plt.subplot(2, 10, x + 1+10)
            plt.imshow(transforms.ToPILImage()(reconstruction[x]))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            plt.savefig('junk.png')

        break

