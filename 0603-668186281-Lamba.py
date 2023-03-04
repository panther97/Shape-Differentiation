#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from PIL import Image


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
#         self.conv4 = nn.Conv2d(128, 256,3,1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(3200, 256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128, 9)
#         self.fc4 = nn.Linear(128,9)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 4)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 4)
        x = self.dropout1(x)
#         x = self.conv4(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
#         x = self.fc3(x)
#         x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = F.softmax(x)
        return x


# In[ ]:


pretrained_model = Net()
pretrained_model.load_state_dict(torch.load('/kaggle/input/geometry-cnn/0603-668186281-Lamba.pt')) #Location of pretrained model 
pretrained_model.eval()


# In[ ]:


classes = ['Circle', 'Heptagon', 'Hexagon', 'Nonagon', 'Octagon', 'Pentagon', 'Square', 'Star', 'Triangle']

transform = transforms.Compose([
transforms.Grayscale(num_output_channels=1),
transforms.ToTensor(),
transforms.Normalize((0.1307,), (0.3081,))])

path_val = '/kaggle/input/validation/val_set'
for i in os.listdir(path_val):
    img = Image.open(os.path.join(path_val,i))
    image = transform(img)
    image = image.unsqueeze(0)
    output = pretrained_model(image)
    print(i,":",classes[output.argmax()])


# In[ ]:




