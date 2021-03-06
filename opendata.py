# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:38:24 2020

@author: Admin
"""
import torch
import numpy as np
import time, os, re
from collections import OrderedDict, defaultdict
import collections
from torch.utils import data

"Get the data"
test_data = open('test_set.txt').read().splitlines()#load data
del test_data[0] #remove the name compostion and delta_e
train_data=open('train_set.txt').read().splitlines()
del train_data[0]

"For processing the data"
def Get_data(Data_input):
    
    Middle_step1=[] #middlestep1
    Middle_step2=[] #middlestep2
    targetvalues=np.zeros(len(Data_input)) #Extract the target values
    name=[] #Extract the compounds
    Valueinput=[] # Extract the numerical value for the elements in the compund

    for i in range(len(Data_input)):
        Middle_step1.append(Data_input[i].split(" ")) #splitting to individual lists
    for j in range(len(Middle_step1)):
        Middle_step2.append(Middle_step1[j][0]) #removing all targets 
        targetvalues[j]=Middle_step1[j][1] #Extracting targetvalue for each input
    
    
    for k in range(len(Middle_step2)):
        Valueinput.append(re.findall('\d*\.?\d+',Middle_step2[k])) #Extracting the numerical value 
        name.append(re.findall('[(a-zA-Z)]',Middle_step2[k])) #Extracting the characters for each input, will be put together later

    for c in range(len(name)):
        name[c][0:len(name[c])]=[''.join(name[c][0:len(name[c])])] #mashing the characters together
    
        
    
    




    elements = ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Sc', 'Ti', 'V', 
            'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 
            'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 
            'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 
            'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu']



# Code to assign the numerical value at the right element index for every compound/input
    formulare = re.compile(r'([A-Z][a-z]*)(\d*)')
    def parse_elements(compound):
        pairs = formulare.findall(compound)
        length = sum((len(p[0]) + len(p[1]) for p in pairs))
        assert length == len(compound)
        compound_dict = defaultdict(int)
        for el, sub in pairs:
            compound_dict[el] += float(sub) if sub else 1
        return compound_dict



    for i in range(len(name)):
        name[i] = [parse_elements(x) for x in name[i]]
    
    
    i = -1

    input = np.zeros(shape=(len(name), 86), dtype=np.float32)
    for j in range(len(name)):
        s=-1
    
        ja=Valueinput[j]
        for compound in name[j]:
            i +=1
            hej=compound
            hej2=hej.keys()
            for k in hej2:
                s +=1
                input[i][elements.index(k)]=ja[s]
            
            
            
    
    
    
    #data = input #training data
    
    X_data=torch.from_numpy(input) #converted to tensor

    Y_data=torch.from_numpy(targetvalues) #convert to tensor
    Y_data=Y_data.reshape(-1,1) #make correct dimension
    
        
    return X_data,Y_data

"Push data thorugh the dataset class in pytorch"
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


"load the dataset into desired batchsizes"
def loaddataset(Set,batchsize,shuff):
    return torch.utils.data.DataLoader(Set,batch_size=batchsize,shuffle=shuff)




    
    

    




    







    








