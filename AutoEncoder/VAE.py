
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
        nn.Conv2d(in_channels=1, out_channels=24, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3)
        )

        self.fc1 = nn.Linear(in_features=96*5*5, out_features=1200)
        self.fc2 = nn.Linear(in_features=1200, out_features=600)
        self.fc_mu = nn.Linear(in_features=600, out_features=20)
        self.fc_logvar = nn.Linear(in_features=600, out_features=20)

        self.conv5 =  nn.Linear(in_features=20, out_features=96*5*5) 


        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=96, out_channels=48, kernel_size=3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=24, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )


    def passToEnoch(self, image):

        t = image

        t = self.encoder(t)
        t = F.relu(self.fc1(t.reshape(100,-1)))
        t = F.relu(self.fc2(t))
        mu = self.fc_mu(t)
        logvar = self.fc_logvar(t)

        return mu, logvar

    def passToDenoch(self, BN):

        t = BN
        t = F.relu(self.conv5(t))
       
        t = t.reshape(100,96,5,5)
        t = self.decoder(t)
        GenImg = t

        return GenImg
    
    def Reparameterise(self, mean, logvar):
       
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mean)
        

    def forward(self, image):

        mu, logvar = self.passToEnoch(image)

        
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
'''

train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=True)
data, label = next(iter(train_loader))
mean = data.mean()
std = data.std()
'''


train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)

network = Model()

optimizer = optim.Adam(network.parameters(), lr=0.0001)


def loss_function(pred, images, mu, logvar):
    BCE = F.binary_cross_entropy(pred, images, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE ,KLD



for epoch in range(20):

    total_loss_1 = 0
    reconstruction_Loss = 0
    KLD_loss_2 = 0


    for batch in train_loader: # Get Batch
        
        images, _ = batch
        
        optimizer.zero_grad()
        construction, mu, logvar = network(images) # Pass Batch
        reconstructionLoss, KLD = loss_function(construction, images, mu, logvar) # Calculate Loss
        loss  = reconstructionLoss + KLD
        
        loss.backward() # Calculate Gradients
        optimizer.step() # Update Weights

        total_loss_1 += loss.item()
        reconstruction_Loss += reconstructionLoss.item()
        KLD_loss_2 += KLD.item()

    print("epoch", epoch, "total_loss_1:", total_loss_1/60000, "reconstruction_loss: ", reconstruction_Loss/60000,"KLD: ", KLD_loss_2/60000)

torch.save(network.state_dict(), 'VAE.pth')


with torch.no_grad():
    network.eval()

    o = np.random.normal(0,1, (100,20))
    o = torch.from_numpy(o).float()
    Train = False
   
    reconstruction = network.passToDenoch(o)

save_image(reconstruction,'Final.png')
Image('Final.png') 


