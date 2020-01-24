# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:18:33 2020

@author: Admin
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable


#Different sizes
input_size=86
hidden_size1=1024
hidden_size2=512
hidden_size3=256
hidden_size4=128
hidden_size5=64
hidden_size6=32
output_size=1   

class Network(nn.Module):
    def __init__(self, input_size, hidden_size1,hidden_size2,hidden_size3,hidden_size4,hidden_size5,hidden_size6, output_size):
        super(Network, self).__init__()
        self.layer1=nn.Sequential(nn.Linear(input_size,hidden_size1),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size1,hidden_size1),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size1,hidden_size1),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size1,hidden_size1),
                                  nn.ReLU(),
                                  nn.Dropout(0.8)
                                  
                                  )
        self.layer2=nn.Sequential(nn.Linear(hidden_size1,hidden_size2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size2,hidden_size2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size2,hidden_size2),
                                  nn.ReLU(),
                                  nn.Dropout(0.9)
                                  )
        self.layer3=nn.Sequential(nn.Linear(hidden_size2,hidden_size3),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size3,hidden_size3),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size3,hidden_size3),
                                  nn.ReLU(),
                                  nn.Dropout(0.7)
                                  )
        self.layer4=nn.Sequential(nn.Linear(hidden_size3,hidden_size4),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size4,hidden_size4),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size4,hidden_size4),
                                  nn.ReLU(),
                                  nn.Dropout(0.8)
                                  )
        self.lastlayer=nn.Sequential(nn.Linear(hidden_size4,hidden_size5),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size5,hidden_size5),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size5,hidden_size6),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size6,output_size)
                                  
                                  )
    def forward(self,x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.lastlayer(out)
        return out