'''

####
DataSet and Results where over 350 MB which was too much for GitHub to accept.
####

'''

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from PIL import Image
import torchvision.transforms.functional as TF


import torchvision
import torchvision.transforms as transforms

import random
from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize


import numpy as np
import pandas as pd


IMG_WIDTH = 572
IMG_HEIGHT = 572
IMG_CHANNELS = 3

TRAIN_PATH = 'stage1_train/'

train_ids = next(os.walk(TRAIN_PATH))[1]



Y_train = torch.zeros((len(train_ids), 1, 388, 388))
X_train = torch.zeros((len(train_ids), IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH))
save = 0

print('Resizing training images and masks')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]  
    
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    img = transforms.ToTensor()(img)
    X_train[n] = img  #Fill empty X_train with values from img
    
    mask = np.zeros((388, 388, 1), dtype=np.bool)

    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (388, 388), mode='constant',  
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)  
        msk = transforms.ToTensor()(mask)
    Y_train[n] = msk

Y_train = Y_train.unsqueeze(1)
X_train = X_train.unsqueeze(1)


print("size of Y is: ", Y_train.size(), "size of X is: ", X_train.size())
print("size of Y is: ", Y_train[0].size(), "size of X is: ", X_train[0].size())



#transforms.ToTensor()
def dice_coeff(pred, target):
    pred = F.softmax(pred,dim=1)
    target[target!=0] = 1
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)



SMOOTH = 1e-6
def IOU_loss(pred, mask):
    #pred = pred.squeeze(1)
    #mask = mask.squeeze(1)
    intersect = (pred*mask).sum(2).sum(1)
    union = (pred+mask).sum(2).sum(1)
    iou = (intersect+0.001)/(union-intersect+0.001)

    return iou.mean()

def double_conv(in_channels, out_channels):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3),
        #nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        
        nn.Conv2d(out_channels, out_channels, kernel_size=3),
        nn.ReLU(inplace=True)
    )

    return conv

def crop_this(tensor, target):
    target_size = target.size()[2]
    tensor_size = tensor.size()[2]
    crop_size = tensor_size - target_size
    crop_size = crop_size // 2
    
    return tensor[:, :, crop_size:tensor_size-crop_size, crop_size:tensor_size-crop_size]

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.Max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_conv1 = double_conv(3,64)
        self.down_conv2 = double_conv(64,128)
        self.down_conv3 = double_conv(128,256)
        self.down_conv4 = double_conv(256,512)
        self.down_conv5 = double_conv(512,1024)

        self.up_convT1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_convT2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_convT3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_convT4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        self.up_conv1 = double_conv(1024,512)
        self.up_conv2 = double_conv(512,256)
        self.up_conv3 = double_conv(256,128)
        self.up_conv4 = double_conv(128,64)

        self.out = nn.Conv2d(64,1, kernel_size=1)

    def forward(self, image):
        

        c1 = self.down_conv1(image)
        c1_pooled = self.Max_pool(c1)

        c2 = self.down_conv2(c1_pooled)
        c2_pooled = self.Max_pool(c2)

        c3 = self.down_conv3(c2_pooled)
        c3_pooled = self.Max_pool(c3)

        c4 = self.down_conv4(c3_pooled)
        c4_pooled = self.Max_pool(c4)

        c5 = self.down_conv5(c4_pooled)

        #return c5

        t1 = self.up_convT1(c5)
        cropped1 = crop_this(c4,t1)
        u1 = self.up_conv1(torch.cat([t1, cropped1],1)) # might be the other way around

        t2 = self.up_convT2(u1)
        cropped2 = crop_this(c3,t2)
        u2 = self.up_conv2(torch.cat([t2, cropped2],1))

        t3 = self.up_convT3(u2)
        cropped3 = crop_this(c2,t3)
        u3 = self.up_conv3(torch.cat([t3, cropped3],1))

        t4 = self.up_convT4(u3)
        cropped4 = crop_this(c1,t4)
        
        u4 = self.up_conv4(torch.cat([t4, cropped4],1))

        final = self.out(u4)
        
        return final




if __name__ == "__main__":
    
    model = UNet()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    
    for epoch in range(1000):
        x = random.randint(0, 669)
        pred = model(X_train[x])
        loss = dice_coeff(pred,Y_train[x])

        optimizer.zero_grad()
        loss.backward() # Calculate Gradients
        optimizer.step() # Update Weights

        torch.save(model.state_dict(), 'third.pth')
       
        
        print("epoch:",epoch," loss:", loss.item())
        

    model.eval()
    for x in range(669):
        #x = random.randint(0, 669)
        pred = model(X_train[x]) 
        plt.imshow(transforms.ToPILImage()(Y_train[x][0]))
        plt.savefig('junk/pic_one/pic_one_'+ str(x) +'.png')
        plt.imshow(transforms.ToPILImage()(X_train[x][0]))
        plt.savefig('junk/pic_two/pic_two_'+ str(x) +'.png')
        plt.imshow(transforms.ToPILImage()(pred[0]))
        plt.savefig('junk/pic_three/pic_three_'+ str(x) +'.png')

    
