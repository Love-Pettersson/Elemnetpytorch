# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:24:09 2020

@author: Admin
"""

import argparse
import os
import sys
import time
import torch.nn as nn
import torch
from torch.optim import Adam
from torch.utils import data
from Thenetwork import Network
from opendata import Get_data,Dataset,loaddataset

parser = argparse.ArgumentParser(description='Elemnet Example')
parser.add_argument('--batch-size', type=int,default=32, metavar='N',
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=4000, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--learningrate', type=int, default=0.0001, metavar='N',
                    help='Size of learningrate')
parser.add_argument('--momentum', type=int, default=0.9, metavar='N',
                    help='Size of momentum parameter')
parser.add_argument('--Evalfreq', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--patience', type=int,default=200, metavar='N',
                    help='number of steps which if test loss has not decreased, training will stop')

args = parser.parse_args()

"Deciding to run on gpu or cpu"
device = torch.device("cuda" if  torch.cuda.is_available() else "cpu") 

"The different layer sizes if the network. Note:some layers share size"
input_size=86
hidden_size1=1024
hidden_size2=512
hidden_size3=256
hidden_size4=128
hidden_size5=64
hidden_size6=32
output_size=1 

"Retrive the model"
model=Network(input_size,hidden_size1,hidden_size2,hidden_size3,hidden_size4,hidden_size5,hidden_size6,output_size).to(device) 

"Get training data"
train_data=open('train_set.txt').read().splitlines() 
del train_data[0] #remove the name compostion and delta_e
X_train=Get_data(train_data)[0] 
Y_train=Get_data(train_data)[1].float() #error if not made float
Train_set=Dataset(X_train,Y_train)
Trainset_loader=loaddataset(Train_set,args.batch_size,True)

"Get test data"
test_data = open('test_set.txt').read().splitlines()
del test_data[0] #remove the name compostion and delta_e
X_test=Get_data(test_data)[0]
Y_test=Get_data(test_data)[1].float() #error if not made float
Test_set=Dataset(X_test,Y_test)
Testset_loader=loaddataset(Test_set,args.batch_size,False)

"Create optimizer aswell as training/test loss function "
Testloss=nn.L1Loss()
Trainingloss=nn.L1Loss()
optimizer=torch.optim.Adam(model.parameters(), lr=args.learningrate)

"Defining variables for the training loop"
best_test_error=100
best_step=0
step=0
patience_steps = int(args.patience * len(X_train)/(args.batch_size))

"Create test function"
def test(epoch): 
    model.eval() #deactivates dropout
    test_loss=0 
    with torch.no_grad(): #no need for gradients in test phase
        k=0
        for i,(inputs,labels) in enumerate(Testset_loader):
            inputs=inputs.to(device)
            labels=labels.to(device)
            output=model(inputs)
            test_loss +=Testloss(output,labels)
            k +=1
    model.train()        
    return test_loss/k 

"Create training function"
def train(epoch):
    model.train()
    global best_test_error #have to made global
    global step
    global best_step
    
    for batch_idx,(inputs,target) in enumerate(Trainset_loader):
        inputs=inputs.to(device)
        target=target.to(device)
        optimizer.zero_grad()
        output=model(inputs)
        loss=Trainingloss(output,target)
        loss.backward()
        optimizer.step()
        step +=1
        
        
        if batch_idx % args.Evalfreq == 0:
            
            Testsetloss=test(epoch)
            if best_test_error>Testsetloss:
                best_test_error=Testsetloss
                best_step=step
                torch.save(model.state_dict(),'checkpoint.pth')
                print('Model saved at: {}'.format('checkpoint.pth'))
            
            print('Train Epoch:{} Minibatch Loss:{}'.format(epoch,loss.item()))
            print('Test set loss: {} Best test loss:{}'.format(test(epoch),best_test_error))
            
"Create training loop"
epoch=1
while epoch < (args.epochs + 1):
    train(epoch)
    if (best_step + patience_steps) <= step:
           print('No improvement in the last {} steps, best test error acheived was {}'.format(patience_steps,best_test_error))
           break
    
    epoch +=1