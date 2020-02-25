# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:53:13 2020

@author: Admin
"""

import sys
import torch.nn as nn
import torch
from torch.optim import AdamW
from torch.utils import data
import time, os, re
from collections import OrderedDict, defaultdict
import collections
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pickle,gzip
from torch.optim.lr_scheduler import StepLR

"Quatenary"
Middle_step1=[]
Elements=[]
Element1=[]
Element2=[]
Element3=[]
Element4=[]

name=[]
with gzip.open('quaternary.pickle.gz', 'rb') as ifp:
    dataset=pickle.load(ifp)
Targetvalues=np.zeros(len(dataset))
for u in range(len(dataset)):
     Elements.append(re.findall('[A-Z][^A-Z]*',dataset[u][1]))

for i in range(len(dataset)):
    Targetvalues[i]=(dataset[i][0]/1000)
    
for s in range(len(dataset)):
    Element1.append(Elements[s][0])
    Element2.append(Elements[s][1])
    result = ''.join([i for i in Elements[s][2] if not i.isdigit()])
    Element3.append(result)
    Element4.append(Elements[s][3])
    
print(Element1[0])
print(Element2[0])
print(Element3[0])
print(Element4[0])

formulare = re.compile(r'([A-Z][a-z]*)(\d*)')
def parse_formula(formula):
    pairs = formulare.findall(formula)
    length = sum((len(p[0]) + len(p[1]) for p in pairs))
    assert length == len(formula)
    formula_dict = defaultdict(int)
    for el, sub in pairs:
        formula_dict[el] += float(sub) if sub else 1
    return formula_dict

formulasA = [parse_formula(x) for x in Element1]
formulasB = [parse_formula(x) for x in Element2]
formulasC = [parse_formula(x) for x in Element3]
formulasD = [parse_formula(x) for x in Element4]
    
    

    
Row_1=['H','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q']
Row_2=['Li', 'Be','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q', 'B', 'C', 'N', 'O', 'F']
Row_3=['Na', 'Mg','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q', 'Al', 'Si', 'P', 'S', 'Cl']
Row_4=['K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br']
Row_5=['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I']
Row_6=['Cs', 'Ba', 'La', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi','Po']

"Representation for A"
Row_input_1A=np.zeros(shape=(len(dataset),17),dtype=np.float32)
Row_input_2A=np.zeros(shape=(len(dataset),17),dtype=np.float32)
Row_input_3A=np.zeros(shape=(len(dataset),17),dtype=np.float32)
Row_input_4A=np.zeros(shape=(len(dataset),17),dtype=np.float32)
Row_input_5A=np.zeros(shape=(len(dataset),17),dtype=np.float32)
Row_input_6A=np.zeros(shape=(len(dataset),17),dtype=np.float32)

"Representation for B"
Row_input_1B=np.zeros(shape=(len(dataset),17),dtype=np.float32)
Row_input_2B=np.zeros(shape=(len(dataset),17),dtype=np.float32)
Row_input_3B=np.zeros(shape=(len(dataset),17),dtype=np.float32)
Row_input_4B=np.zeros(shape=(len(dataset),17),dtype=np.float32)
Row_input_5B=np.zeros(shape=(len(dataset),17),dtype=np.float32)
Row_input_6B=np.zeros(shape=(len(dataset),17),dtype=np.float32)

"Representation for C"
Row_input_1C=np.zeros(shape=(len(dataset),17),dtype=np.float32)
Row_input_2C=np.zeros(shape=(len(dataset),17),dtype=np.float32)
Row_input_3C=np.zeros(shape=(len(dataset),17),dtype=np.float32)
Row_input_4C=np.zeros(shape=(len(dataset),17),dtype=np.float32)
Row_input_5C=np.zeros(shape=(len(dataset),17),dtype=np.float32)
Row_input_6C=np.zeros(shape=(len(dataset),17),dtype=np.float32)

"Representation for D"
Row_input_1D=np.zeros(shape=(len(dataset),17),dtype=np.float32)
Row_input_2D=np.zeros(shape=(len(dataset),17),dtype=np.float32)
Row_input_3D=np.zeros(shape=(len(dataset),17),dtype=np.float32)
Row_input_4D=np.zeros(shape=(len(dataset),17),dtype=np.float32)
Row_input_5D=np.zeros(shape=(len(dataset),17),dtype=np.float32)
Row_input_6D=np.zeros(shape=(len(dataset),17),dtype=np.float32)

Input_final=np.zeros(shape=(len(dataset),4,6,17),dtype=np.float32)


i = -1
for formula in formulasA:
    i+=1
    keys = formula.keys()
    for k in keys:
        if k=='H':
            Row_input_1A[i][Row_1.index(k)] = 160
        if any ([k=='Li',k=='Be',k=='B',k=='C',k=='N',k=='O',k=='F']):
            Row_input_2A[i][Row_2.index(k)] = 160
        if any ([k=='Na',k=='Mg',k=='Al',k=='Si',k=='P',k=='S',k=='Cl']):
            Row_input_3A[i][Row_3.index(k)] = 160
        if any ([k=='K',k=='Ca',k=='Sc',k=='Ti',k=='V',k=='Cr',k=='Mn',k=='Fe',k=='Co',k=='Ni',k=='Cu',k=='Zn',k=='Ga',k=='Ge',k=='As',k=='Se',k=='Br']):
            Row_input_4A[i][Row_4.index(k)] = 160
        if any ([k=='Rb',k=='Sr',k=='Y',k=='Zr',k=='Nb',k=='Mo',k=='Tc',k=='Ru',k=='Rh',k=='Pd',k=='Ag',k=='Cd',k=='In',k=='Sn',k=='Sb',k=='Te',k=='I']):
            Row_input_5A[i][Row_5.index(k)] = 160
        if any ([k=='Cs',k=='Ba',k=='La',k=='Lu',k=='Hf',k=='Ta',k=='W',k=='Re',k=='Os',k=='Ir',k=='Pt',k=='Au',k=='Hg',k=='Tl',k=='Pb',k=='Bi']):
            Row_input_6A[i][Row_6.index(k)] = 160
            
j = -1           
for formula in formulasB:
    j+=1
    keys = formula.keys()
    for k in keys:
        if k=='H':
            Row_input_1B[j][Row_1.index(k)] = 160
        if any ([k=='Li',k=='Be',k=='B',k=='C',k=='N',k=='O',k=='F']):
            Row_input_2B[j][Row_2.index(k)] = 160
        if any ([k=='Na',k=='Mg',k=='Al',k=='Si',k=='P',k=='S',k=='Cl']):
            Row_input_3B[j][Row_3.index(k)] = 160
        if any ([k=='K',k=='Ca',k=='Sc',k=='Ti',k=='V',k=='Cr',k=='Mn',k=='Fe',k=='Co',k=='Ni',k=='Cu',k=='Zn',k=='Ga',k=='Ge',k=='As',k=='Se',k=='Br']):
            Row_input_4B[j][Row_4.index(k)] = 160
        if any ([k=='Rb',k=='Sr',k=='Y',k=='Zr',k=='Nb',k=='Mo',k=='Tc',k=='Ru',k=='Rh',k=='Pd',k=='Ag',k=='Cd',k=='In',k=='Sn',k=='Sb',k=='Te',k=='I']):
            Row_input_5B[j][Row_5.index(k)] = 160
        if any ([k=='Cs',k=='Ba',k=='La',k=='Lu',k=='Hf',k=='Ta',k=='W',k=='Re',k=='Os',k=='Ir',k=='Pt',k=='Au',k=='Hg',k=='Tl',k=='Pb',k=='Bi']):
            Row_input_6B[j][Row_6.index(k)] = 160
s = -1
for formula in formulasC:
    s+=1
    keys = formula.keys()
    for k in keys:
        if k=='H':
            Row_input_1C[s][Row_1.index(k)] = 320
        if any ([k=='Li',k=='Be',k=='B',k=='C',k=='N',k=='O',k=='F']):
            Row_input_2C[s][Row_2.index(k)] = 320
        if any ([k=='Na',k=='Mg',k=='Al',k=='Si',k=='P',k=='S',k=='Cl']):
            Row_input_3C[s][Row_3.index(k)] = 320
        if any ([k=='K',k=='Ca',k=='Sc',k=='Ti',k=='V',k=='Cr',k=='Mn',k=='Fe',k=='Co',k=='Ni',k=='Cu',k=='Zn',k=='Ga',k=='Ge',k=='As',k=='Se',k=='Br']):
            Row_input_4C[s][Row_4.index(k)] = 320
        if any ([k=='Rb',k=='Sr',k=='Y',k=='Zr',k=='Nb',k=='Mo',k=='Tc',k=='Ru',k=='Rh',k=='Pd',k=='Ag',k=='Cd',k=='In',k=='Sn',k=='Sb',k=='Te',k=='I']):
            Row_input_5C[s][Row_5.index(k)] = 320
        if any ([k=='Cs',k=='Ba',k=='La',k=='Lu',k=='Hf',k=='Ta',k=='W',k=='Re',k=='Os',k=='Ir',k=='Pt',k=='Au',k=='Hg',k=='Tl',k=='Pb',k=='Bi']):
            Row_input_6C[s][Row_6.index(k)] = 320
            
a=-1 
for formula in formulasD:
    a+=1
    keys = formula.keys()
    for k in keys:
        if k=='H':
            Row_input_1D[a][Row_1.index(k)] = 160
        if any ([k=='Li',k=='Be',k=='B',k=='C',k=='N',k=='O',k=='F']):
            Row_input_2D[a][Row_2.index(k)] = 160
        if any ([k=='Na',k=='Mg',k=='Al',k=='Si',k=='P',k=='S',k=='Cl']):
            Row_input_3D[a][Row_3.index(k)] = 160
        if any ([k=='K',k=='Ca',k=='Sc',k=='Ti',k=='V',k=='Cr',k=='Mn',k=='Fe',k=='Co',k=='Ni',k=='Cu',k=='Zn',k=='Ga',k=='Ge',k=='As',k=='Se',k=='Br']):
            Row_input_4D[a][Row_4.index(k)] = 160
        if any ([k=='Rb',k=='Sr',k=='Y',k=='Zr',k=='Nb',k=='Mo',k=='Tc',k=='Ru',k=='Rh',k=='Pd',k=='Ag',k=='Cd',k=='In',k=='Sn',k=='Sb',k=='Te',k=='I']):
            Row_input_5D[a][Row_5.index(k)] = 160
        if any ([k=='Cs',k=='Ba',k=='La',k=='Lu',k=='Hf',k=='Ta',k=='W',k=='Re',k=='Os',k=='Ir',k=='Pt',k=='Au',k=='Hg',k=='Tl',k=='Pb',k=='Bi']):
            Row_input_6D[a][Row_6.index(k)] = 160



for f in range(len(dataset)):
    Input_final[f][0][0]=Row_input_1A[f]
    Input_final[f][0][1]=Row_input_2A[f]
    Input_final[f][0][2]=Row_input_3A[f]
    Input_final[f][0][3]=Row_input_4A[f]
    Input_final[f][0][4]=Row_input_5A[f]
    Input_final[f][0][5]=Row_input_6A[f]

    Input_final[f][1][0]=Row_input_1B[f]
    Input_final[f][1][1]=Row_input_2B[f]
    Input_final[f][1][2]=Row_input_3B[f]
    Input_final[f][1][3]=Row_input_4B[f]
    Input_final[f][1][4]=Row_input_5B[f]
    Input_final[f][1][5]=Row_input_6B[f]

    Input_final[f][2][0]=Row_input_1C[f]
    Input_final[f][2][1]=Row_input_2C[f]
    Input_final[f][2][2]=Row_input_3C[f]
    Input_final[f][2][3]=Row_input_4C[f]
    Input_final[f][2][4]=Row_input_5C[f]
    Input_final[f][2][5]=Row_input_6C[f]

    Input_final[f][3][0]=Row_input_1D[f]
    Input_final[f][3][1]=Row_input_2D[f]
    Input_final[f][3][2]=Row_input_3D[f]
    Input_final[f][3][3]=Row_input_4D[f]
    Input_final[f][3][4]=Row_input_5D[f]
    Input_final[f][3][5]=Row_input_6D[f]
 
class Dataset(data.Dataset):
  
  def __init__(self, inputvector, labels):
        
        self.labels = labels
        self.inputvector = inputvector
        
        
  def __len__(self):
        
        return len(self.inputvector)

  def __getitem__(self, index):
      
      X=self.inputvector[index]
      Y=self.labels[index]
      
      return X,Y
  
"Reshaping targetvalues, and randomly split the dataset into train and test"
Targetvalues=Targetvalues.reshape(-1,1)
Targetvalues=torch.from_numpy(Targetvalues)
Input_final=torch.from_numpy(Input_final)
the_data=Dataset(Input_final,Targetvalues)
train_dataset, test_dataset,rest = torch.utils.data.random_split(the_data, [160000,26400,71]) #batch_size is hundred thus receive an error for the 71 left


"Creating the network"
class Network(nn.Module):
    def __init__(self, hidden_size1,hidden_size2,hidden_size3,hidden_size4,hidden_size5,hidden_size6, output_size):
        super(Network, self).__init__()
        self.CNNlayer=nn.Sequential(nn.Conv3d(1,92,(1,6,3),1,0),nn.ReLU()) #92 filters, 2D-filter worked better than 3D 
        self.reshape=nn.Sequential(nn.Linear(5520,hidden_size1),nn.ReLU())
        self.layer1=nn.Sequential(
                                  nn.Linear(hidden_size1,hidden_size1),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size1,hidden_size1),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size1,hidden_size1),
                                  nn.ReLU()
                                  
                                  )
        self.layer2=nn.Sequential(nn.Linear(hidden_size1,hidden_size2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size2,hidden_size2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size2,hidden_size2),
                                  nn.ReLU()
                                  )
        self.layer3=nn.Sequential(nn.Linear(hidden_size2,hidden_size3),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size3,hidden_size3),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size3,hidden_size3),
                                  nn.ReLU()
                                  )
        self.layer4=nn.Sequential(nn.Linear(hidden_size3,hidden_size4),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size4,hidden_size4),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size4,hidden_size4),
                                  nn.ReLU()
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
        x = x.view(100, 1, 4, 6, 17) #batch_size, in_channels, width and height of kernel
        out=self.CNNlayer(x)
        out=out.view(out.size(0),-1) #flatten it to one dim
        out=self.reshape(out)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.lastlayer(out)
        return out


"batch_size, learningrate etc."   
batchsize=100
learningrate=0.001
patience=100
Evalfreq=1
epochs=1000
"Create summary writer"
#writer = SummaryWriter(log_dir='/content/drive/My Drive/Colab Notebooks/Tensorboardsave/Documentation',filename_suffix='quaternary')

"Deciding to run on gpu or cpu"
device = torch.device("cuda" if  torch.cuda.is_available() else "cpu") 

train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batchsize,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batchsize,shuffle=False)

"The different layer sizes if the network. Note:some layers share size"
 
hidden_size1=1024
hidden_size2=512
hidden_size3=256
hidden_size4=128
hidden_size5=64
hidden_size6=32
output_size=1 

"Retrive the model"
model=Network(hidden_size1,hidden_size2,hidden_size3,hidden_size4,hidden_size5,hidden_size6,output_size).to(device) 



"Create optimizer aswell as training/test loss function "
Testloss=nn.L1Loss()
Trainingloss=nn.L1Loss()
optimizer=torch.optim.AdamW(model.parameters(), lr=learningrate)
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

"Defining variables for the training loop"
best_test_error=100
best_step=0
step=0


"Create test function"
def test(epoch): 
    model.eval() 
    test_loss=0
    k=0 
    with torch.no_grad(): #no need for gradients in test phase
        
        for i,(inputs,labels) in enumerate(test_loader):
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
    k=0
    Totloss=0 #Show accumalted loss, not just over one minibatch
    
    for batch_idx,(inputs,target) in enumerate(train_loader):
        
        inputs=inputs.to(device)
        target=target.to(device)
        optimizer.zero_grad()
        output=model(inputs)
        loss=Trainingloss(output,target)
        loss.backward()
        optimizer.step()
        Totloss +=loss.item()
        k +=1  

   
 
    
    print('Train Epoch:{} Training Loss:{}'.format(epoch,Totloss/k))
    #writer.add_scalar('training_loss',Totloss/k,epoch)
        
       
            
"Create training loop"
epoch=0
while epoch < (epochs + 1):
    train(epoch)
    scheduler.step()
    step +=1
    print('LR:', scheduler.get_lr())
    if epoch%Evalfreq==0:
       Testsetloss=test(epoch)
       if best_test_error > Testsetloss:
          best_test_error=Testsetloss
          best_step=step
          #torch.save(model.state_dict(),'/content/drive/My Drive/Colab Notebooks/quaternary.pth')
       print('Test set loss: {} Best test loss:{}'.format(Testsetloss,best_test_error))
       #writer.add_scalar('test_loss',Testsetloss,epoch)

    if (best_step + patience) <= step:
           print('No improvement in the last {} steps, best test error acheived was {}'.format(patience,best_test_error))
           print('Done!')
           break
    
    epoch +=1
#writer.close()