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

from torchvision.utils import save_image
from IPython.display import Image

batch_size = 100 #- 500
epochs = 20
learning_rate = 1e-3

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 

device = torch.device(dev)

Train = True

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784,20**2),
            nn.ReLU(),
            nn.Linear(20**2,20*2)
        )

        self.fc_mu = nn.Linear(in_features=20*2, out_features=20)
        self.fc_logvar = nn.Linear(in_features=20*2, out_features=20)

        self.decoder = nn.Sequential(
            nn.Linear(20,20**2),
            nn.ReLU(),
            nn.Linear(20**2,784),
            nn.Sigmoid()
        )

    def passToEnoch(self, image):

        bottleNeck = self.enoch(image)

        return bottleNeck

    def passToDenoch(self, BN):

        GenImg = self.decoder(BN)

        return GenImg
    
    def Reparameterise(self, mean, logvar):
       
        #std = logvar.mul(0.5).exp_()
        #eps = std.data.new(std.size()).normal_()
        #return eps.mul(std).add_(mean)
        
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        #return eps * torch.exp(logvar * .5) + mean
        return eps.mul(std).add_(mean)
        

    def forward(self, image):

        output = self.encoder(image.reshape(-1,784))

        mu = self.fc_mu(output)
        logvar =  self.fc_logvar(output)

        z = self.Reparameterise(mu, logvar)
        
        construction = self.passToDenoch(z)
        
        return construction, mu, logvar

train_set = torchvision.datasets.MNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)


train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=True)
data, label = next(iter(train_loader))
mean = data.mean()
std = data.std()



train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)

network = Model()
#network = network.to(torch.device("cuda:0"))

optimizer = optim.Adam(network.parameters(), lr=0.001)


def loss_function(pred, images, mu, logvar):

    criterion = nn.BCELoss(reduction = 'sum')
    
    reconstructionLoss = criterion(pred,images.reshape(-1,784))

    KLD  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return reconstructionLoss , KLD



for epoch in range(20):

    total_loss_1 = 0
    reconstruction_loss = 0
    kld_loss = 0


    for batch in train_loader: # Get Batch
        
        images, _ = batch
        #images = (images - mean) / (std + 1e-15)
        #images = images.to(torch.device("cuda:0"))
        
        construction, mu, logvar = network(images) # Pass Batch
        reconstructionLoss , KLD = loss_function(construction, images, mu, logvar) # Calculate Loss
        loss = reconstructionLoss + KLD
        optimizer.zero_grad()
        loss.backward() # Calculate Gradients
        optimizer.step() # Update Weights

        total_loss_1 += loss.item()
        reconstruction_loss += reconstructionLoss.item()
        kld_loss += KLD.item()

    print("epoch", epoch, "total_loss_1:", total_loss_1/60000, "reconstruction_loss: ", reconstruction_loss/60000,"KLD: ", kld_loss/60000)

torch.save(network.state_dict(), 'fc_VAE.pth')

#model = torch.load('VAE.pth')


with torch.no_grad():
    network.eval()

    o = np.random.normal(0,1, (100,20))
    o = torch.from_numpy(o).float()
    Train = False
    
    reconstruction = network.passToDenoch(o).reshape(100,1,28,28)

    save_image(reconstruction,'Final_fc.png')
    Image('Final_fc.png') 

