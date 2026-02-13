#LeNet-5 copy, meant to be trained on MNIST dataset
#learning excercise for pytorch, convolutional and linear layers
#Pete Zimerman
#Feb-2026
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F

#combined convolutional first layer with mlp to give output
#1x convolutional layer, nx linear layers
class LeCopy(torch.nn.Module):

    def __init__(self):
        super().__init__()

        #convolutional layers
        self.con1 = torch.nn.Conv2d(1, 6, 5)
        self.con2 = torch.nn.Conv2d(6, 16, 6)

        #linear layers
        self.lin1 = torch.nn.Linear(16*3*3, 120)
        self.lin2 = torch.nn.Linear(120, 84)
        self.output = torch.nn.Linear(84,10)
    
    #feed forward
    def forward(self, image):
        #pass through convolutional layers
        image = image
        image = F.max_pool2d(F.relu(self.con1(image)) , 2)
        image = F.max_pool2d(F.relu(self.con2(image)), 2)

        #reformat into 1d tensor
        image = torch.flatten(image,1)

        #pass through linear layers
        image = F.relu(self.lin1(image))
        image = F.relu(self.lin2(image))
        return self.output(image)