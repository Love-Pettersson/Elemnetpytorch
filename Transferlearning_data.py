# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 22:07:27 2020

@author: Admin
"""

import numpy as np
import time, os, re
from collections import OrderedDict, defaultdict
import collections
import pickle,gzip
from torch.utils import data
import torch
import torch.nn as nn

test1=[]
test2=[]
test3=[]
Elements=[] 
A=[]
B=[]
X=[]
hej=[]
hej2=[]

test_data = open('energy').read().splitlines()
test_data2= open('volume').read().splitlines()
test_data3= open('gaps').read().splitlines()

targetvalues_formenergy=np.zeros(len(test_data))
targetvalues_gaps=np.zeros(len(test_data3))
targetvalues_volume=np.zeros(len(test_data2))

for i in range(len(test_data)):
    hej.append(test_data[i].split())


for j in range(len(hej)):
    targetvalues_formenergy[j]=hej[j][2]
    targetvalues_volume[j]=(test_data2[j].split())[2]
    targetvalues_gaps[j]=(test_data3[j].split())[1]
    Elements.append(re.findall('[A-Z][^A-Z]*',hej[j][0]))
    

    
for s in range(len(hej)):
    
    del Elements[s][0]
    hej2.append(Elements[s][2].split('/'))
    Elements[s][2]=hej2[s][0]
    test1.append(re.findall('([A-Z][a-z]*)(\d*)',Elements[s][0]))
    test2.append(re.findall('([A-Z][a-z]*)(\d*)',Elements[s][1]))
    test3.append(re.findall('([A-Z][a-z]*)(\d*)',Elements[s][2]))
    Elements[s][0]=test1[s]
    Elements[s][1]=test2[s]
    Elements[s][2]=test3[s]
    
    

for k in range(len(hej)):
    if Elements[k][0][0][1]=='3':
        Elements[k][0],Elements[k][2]=Elements[k][2],Elements[k][0]
    if Elements[k][1][0][1]=='3':
        Elements[k][1],Elements[k][2]=Elements[k][2],Elements[k][1]

for j in range(len(hej)):
    if (hej2[j][1]=='xxx_02p-00_spg221a' and test1[j][0][1]=='3') :
        Elements[j][0],Elements[j][1]=Elements[j][1],Elements[j][0]
    if (hej2[j][1]=='xxx_02p-00_spg221b' and test2[j][0][1]=='3') :
        Elements[j][0],Elements[j][1]=Elements[j][1],Elements[j][0]
    if (hej2[j][1]=='xxx_02p-00_spg221b' and test3[j][0][1]=='3'):
        Elements[j][0],Elements[j][1]=Elements[j][1],Elements[j][0]
        
        
    

for u in range(len(hej)):
    A.append(Elements[u][0][0][0])
    B.append(Elements[u][1][0][0])
    X.append(Elements[u][2][0][0])
        
formulare = re.compile(r'([A-Z][a-z]*)(\d*)')
def parse_formula(formula):
    pairs = formulare.findall(formula)
    length = sum((len(p[0]) + len(p[1]) for p in pairs))
    assert length == len(formula)
    formula_dict = defaultdict(int)
    for el, sub in pairs:
        formula_dict[el] += float(sub) if sub else 1
    return formula_dict

formulasA = [parse_formula(x) for x in A]
formulasB = [parse_formula(x) for x in B]
formulasC = [parse_formula(x) for x in X]
u=-1
for formula in formulasA:
    u+=1
    keys = formula.keys()
    for k in keys:
        if formula[k]==3:
            formulasA[u],formulasC[u]=formulasC[u],formulasA[u]

p=-1
for formula in formulasB:
    p+=1
    keys = formula.keys()
    for k in keys:
        if formula[k]==3:
            formulasB[p],formulasC[p]=formulasC[p],formulasB[p]
            
    

Row_1=['H','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q']
Row_2=['Li', 'Be','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q', 'B', 'C', 'N', 'O', 'F']
Row_3=['Na', 'Mg','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q', 'Al', 'Si', 'P', 'S', 'Cl']
Row_4=['K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br']
Row_5=['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I']
Row_6=['Cs', 'Ba', 'La', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi','Po']

"Representation for A"
Row_input_1A=np.zeros(shape=(len(test_data),17),dtype=np.float32)
Row_input_2A=np.zeros(shape=(len(test_data),17),dtype=np.float32)
Row_input_3A=np.zeros(shape=(len(test_data),17),dtype=np.float32)
Row_input_4A=np.zeros(shape=(len(test_data),17),dtype=np.float32)
Row_input_5A=np.zeros(shape=(len(test_data),17),dtype=np.float32)
Row_input_6A=np.zeros(shape=(len(test_data),17),dtype=np.float32)

"Representation for B"
Row_input_1B=np.zeros(shape=(len(test_data),17),dtype=np.float32)
Row_input_2B=np.zeros(shape=(len(test_data),17),dtype=np.float32)
Row_input_3B=np.zeros(shape=(len(test_data),17),dtype=np.float32)
Row_input_4B=np.zeros(shape=(len(test_data),17),dtype=np.float32)
Row_input_5B=np.zeros(shape=(len(test_data),17),dtype=np.float32)
Row_input_6B=np.zeros(shape=(len(test_data),17),dtype=np.float32)

"Representation for C"
Row_input_1C=np.zeros(shape=(len(test_data),17),dtype=np.float32)
Row_input_2C=np.zeros(shape=(len(test_data),17),dtype=np.float32)
Row_input_3C=np.zeros(shape=(len(test_data),17),dtype=np.float32)
Row_input_4C=np.zeros(shape=(len(test_data),17),dtype=np.float32)
Row_input_5C=np.zeros(shape=(len(test_data),17),dtype=np.float32)
Row_input_6C=np.zeros(shape=(len(test_data),17),dtype=np.float32)


Input_final=np.zeros(shape=(len(test_data),3,6,17),dtype=np.float32)
i = -1
for formula in formulasA:
    i+=1
    keys = formula.keys()
    
    for k in keys:
        if k=='H':
            Row_input_1A[i][Row_1.index(k)] = formula[k]*160
        if any ([k=='Li',k=='Be',k=='B',k=='C',k=='N',k=='O',k=='F']):
            Row_input_2A[i][Row_2.index(k)] = formula[k]*160
        if any ([k=='Na',k=='Mg',k=='Al',k=='Si',k=='P',k=='S',k=='Cl']):
            Row_input_3A[i][Row_3.index(k)] = formula[k]*160
        if any ([k=='K',k=='Ca',k=='Sc',k=='Ti',k=='V',k=='Cr',k=='Mn',k=='Fe',k=='Co',k=='Ni',k=='Cu',k=='Zn',k=='Ga',k=='Ge',k=='As',k=='Se',k=='Br']):
            Row_input_4A[i][Row_4.index(k)] = formula[k]*160
        if any ([k=='Rb',k=='Sr',k=='Y',k=='Zr',k=='Nb',k=='Mo',k=='Tc',k=='Ru',k=='Rh',k=='Pd',k=='Ag',k=='Cd',k=='In',k=='Sn',k=='Sb',k=='Te',k=='I']):
            Row_input_5A[i][Row_5.index(k)] = formula[k]*160
        if any ([k=='Cs',k=='Ba',k=='La',k=='Lu',k=='Hf',k=='Ta',k=='W',k=='Re',k=='Os',k=='Ir',k=='Pt',k=='Au',k=='Hg',k=='Tl',k=='Pb',k=='Bi']):
            Row_input_6A[i][Row_6.index(k)] = formula[k]*160
j = -1
for formula in formulasB:
    j+=1
    keys = formula.keys()
    for k in keys:
        if k=='H':
            Row_input_1B[j][Row_1.index(k)] = 160*formula[k]
        if any ([k=='Li',k=='Be',k=='B',k=='C',k=='N',k=='O',k=='F']):
            Row_input_2B[j][Row_2.index(k)] = 160*formula[k]
        if any ([k=='Na',k=='Mg',k=='Al',k=='Si',k=='P',k=='S',k=='Cl']):
            Row_input_3B[j][Row_3.index(k)] = 160*formula[k]
        if any ([k=='K',k=='Ca',k=='Sc',k=='Ti',k=='V',k=='Cr',k=='Mn',k=='Fe',k=='Co',k=='Ni',k=='Cu',k=='Zn',k=='Ga',k=='Ge',k=='As',k=='Se',k=='Br']):
            Row_input_4B[j][Row_4.index(k)] = 160*formula[k]
        if any ([k=='Rb',k=='Sr',k=='Y',k=='Zr',k=='Nb',k=='Mo',k=='Tc',k=='Ru',k=='Rh',k=='Pd',k=='Ag',k=='Cd',k=='In',k=='Sn',k=='Sb',k=='Te',k=='I']):
            Row_input_5B[j][Row_5.index(k)] = 160*formula[k]
        if any ([k=='Cs',k=='Ba',k=='La',k=='Lu',k=='Hf',k=='Ta',k=='W',k=='Re',k=='Os',k=='Ir',k=='Pt',k=='Au',k=='Hg',k=='Tl',k=='Pb',k=='Bi']):
            Row_input_6B[j][Row_6.index(k)] = 160*formula[k]
s = -1
for formula in formulasC:
    s+=1
    keys = formula.keys()
    for k in keys:
        if k=='H':
            Row_input_1C[s][Row_1.index(k)] = 160*3
        if any ([k=='Li',k=='Be',k=='B',k=='C',k=='N',k=='O',k=='F']):
            Row_input_2C[s][Row_2.index(k)] = 160*3
        if any ([k=='Na',k=='Mg',k=='Al',k=='Si',k=='P',k=='S',k=='Cl']):
            Row_input_3C[s][Row_3.index(k)] = 160*3
        if any ([k=='K',k=='Ca',k=='Sc',k=='Ti',k=='V',k=='Cr',k=='Mn',k=='Fe',k=='Co',k=='Ni',k=='Cu',k=='Zn',k=='Ga',k=='Ge',k=='As',k=='Se',k=='Br']):
            Row_input_4C[s][Row_4.index(k)] = 160*3
        if any ([k=='Rb',k=='Sr',k=='Y',k=='Zr',k=='Nb',k=='Mo',k=='Tc',k=='Ru',k=='Rh',k=='Pd',k=='Ag',k=='Cd',k=='In',k=='Sn',k=='Sb',k=='Te',k=='I']):
            Row_input_5C[s][Row_5.index(k)] = 160*3
        if any ([k=='Cs',k=='Ba',k=='La',k=='Lu',k=='Hf',k=='Ta',k=='W',k=='Re',k=='Os',k=='Ir',k=='Pt',k=='Au',k=='Hg',k=='Tl',k=='Pb',k=='Bi']):
            Row_input_6C[s][Row_6.index(k)] = 160*3
for f in range(len(test_data)):
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
targetvalues_formenergy=targetvalues_formenergy.reshape(-1,1)
targetvalues_formenergy=torch.from_numpy(targetvalues_formenergy)
Input_final=torch.from_numpy(Input_final)
the_data=Dataset(Input_final,targetvalues_formenergy)

