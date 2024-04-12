#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 13:13:19 2022

@author: antonio
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor as GBR
from tqdm import tqdm
# In[]

PATH = ''
Vars = ['Distance','RelErrorBurnup','RelErrorCt']
Data = []
X = []
for v in Vars:
    Data.append(np.load(PATH+'Y{}.npy'.format(v),allow_pickle=True).reshape(-1,1))
    X.append(np.load(PATH+'X{}.npy'.format(v),allow_pickle=True))
Data = np.hstack(Data)
X = X[0]

Xdiv = np.load(PATH+'XKlDiv.npy',allow_pickle=True)
Datadv = np.load(PATH+'YKlDiv.npy',allow_pickle=True)


# In[]
M = np.mean(Data,axis=0)
S = np.std(Data,axis=0)

Data = (Data-M)/S
# In[]

Mx = np.mean(X,axis=0)
Sx = np.std(X,axis=0)

X = (X-Mx)/Sx

# In[]
depth = np.arange(1,101)
R2 = []
for d in tqdm(depth):
    GB = GBR(n_estimators=500,max_depth=d)
    GB.fit(X,Data[:,0])
    R2.append(GB.score(X,Data[:,0]))

# In[]

GB_Distance = GBR(n_estimators=500,max_depth=8)
GB_Distance.fit(X,Data[:,0])
print(GB_Distance.score(X,Data[:,0]))
print(GB_Distance.feature_importances_)
Ypred = GB_Distance.predict(X)

GB_EBurn = GBR(n_estimators=500,max_depth=8)
GB_EBurn.fit(X,Data[:,1])
print(GB_EBurn.score(X,Data[:,1]))
print(GB_EBurn.feature_importances_)
Ypred = GB_EBurn.predict(X)

GB_ET = GBR(n_estimators=500,max_depth=8)
GB_ET.fit(X,Data[:,2])
print(GB_ET.score(X,Data[:,2]))
print(GB_ET.feature_importances_)
Ypred = GB_ET.predict(X)

GB_Div = GBR(n_estimators=500,max_depth=8)
GB_Div.fit(Xdiv,Datadv)
print(GB_Div.score(Xdiv,Datadv))
print(GB_Div.feature_importances_)
Ypred = GB_Div.predict(Xdiv)


# plt.figure()
# plt.scatter(X[:,-1],Data[:,0])
# plt.scatter(X[:,-1],Ypred)

# In[]
##########################################################
SimulationPoints = np.load('/home/antonio/Desktop/Bayesian Inference SA/Sensitivity Analysis New/FirstOrderSamples.npy',allow_pickle=True)


Output = {}
for i in range(SimulationPoints.shape[0]):
    C = SimulationPoints[i]
    Res = np.vstack((GB_Distance.predict(C),GB_EBurn.predict(C),
                     GB_ET.predict(C),GB_Div.predict(C)))
    Output['C{}'.format(int(i-2))] = Res.T


np.save('/home/antonio/Desktop/Bayesian Inference SA/Sensitivity Analysis New/FirstOrderSamples-Results.npy',Output)
###########################################################
SimulationPoints = np.load('/home/antonio/Desktop/Bayesian Inference SA/Sensitivity Analysis New/SecondOrderSamples.npy',allow_pickle=True)


Output = {}
for i in range(SimulationPoints.shape[0]):
    C = SimulationPoints[i]
    Res = np.vstack((GB_Distance.predict(C),GB_EBurn.predict(C),
                     GB_ET.predict(C),GB_Div.predict(C)))
    Output['C{}'.format(int(i-2))] = Res.T


np.save('/home/antonio/Desktop/Bayesian Inference SA/Sensitivity Analysis New/SecondOrderSamples-Results.npy',Output)
        

# In[]
def get_Sensitivity_indexes(Targetidx,Variables,FirstOrderMatrices,SecondOrderMatrices):
    def nsp(x,y):
           if len(x)!=len(y): 
               raise Exception('Warning: Dimensional Error')            
           s=0
           for i in range(len(x)):
               s+=x[i]*y[i]
           return s/len(x)
    f0= np.mean(np.concatenate((FirstOrderMatrices[0][:,Targetidx],FirstOrderMatrices[1][:,Targetidx])))
    Denom = np.var(np.concatenate((FirstOrderMatrices[0][:,Targetidx],FirstOrderMatrices[1][:,Targetidx])))
    Variables = ['A','B'] + Variables
    if np.isnan(Denom):
        print('The Calculation has a problem!!')

    SecondOrderNames = [Variables[i]+'-'+Variables[j] for i in range(2,len(Variables))\
                        for j in range(i+1,len(Variables))]
    SecondOrderNames = ['A','B'] + SecondOrderNames
    FirstOrderIndexes = {}
    for i in range(2,len(FirstOrderMatrices)):

        FirstOrderIndexes[Variables[i]] = (nsp(FirstOrderMatrices[0][:,Targetidx],FirstOrderMatrices[i][:,Targetidx]) \
               - (f0**2))/Denom
        
    SecondOrderIndexes = {}
    for i in range(2,len(SecondOrderMatrices)):
        print(SecondOrderNames[i])
        v1,v2 = SecondOrderNames[i].split('-')
        i1 = Variables.index(v1)
        i2 = Variables.index(v2)
        SecondOrderIndexes[SecondOrderNames[i]] =\
            ((nsp(SecondOrderMatrices[0][:,Targetidx],SecondOrderMatrices[i][:,Targetidx]) \
               - (f0**2))/Denom) - FirstOrderIndexes[v1] - FirstOrderIndexes[v2]
        
    
    return FirstOrderIndexes,SecondOrderIndexes

InputsFirstOrder = np.load('FirstOrderSamples-Results.npy',allow_pickle=True).item()
InputsFirstOrder = [InputsFirstOrder[i] for i in InputsFirstOrder]
InputsSecondOrder = np.load('SecondOrderSamples-Results.npy',allow_pickle=True).item()
InputsSecondOrder = [InputsSecondOrder[i] for i in InputsSecondOrder]

Metrics = ['Distance','RelErrorBurnup','RelErrorCtime','KlDiv']
Vars = ['Burnup','Ctime','NRatios','Uncertainty','Prior Type']

Distance_Indexes = get_Sensitivity_indexes(0, Vars, InputsFirstOrder, InputsSecondOrder)
Burnup_Error_Indexes = get_Sensitivity_indexes(1, Vars, InputsFirstOrder, InputsSecondOrder)
Ctime_Error_Indexes = get_Sensitivity_indexes(2, Vars, InputsFirstOrder, InputsSecondOrder)
Kl_Indexes = get_Sensitivity_indexes(3, Vars, InputsFirstOrder, InputsSecondOrder)
# SumFirstAndSecond = {}
# for metric in Metrics:
#     SumFirstAndSecond[metric] = {}
#     for i in range(len(Vars)):
#         SumFirstAndSecond[metric][Vars[i]] = FirstOrderIndexes[metric][i] +\
#            sum([SecondOrderIndexes[metric][Indices.index(j)] for j in Indices if str(i) in j])

# In[]

InputsFirstOrder = [InputsFirstOrder[i] for i in InputsFirstOrder]

Metrics = ['Distance','RelErrorBurnup','RelErrorCtime','KlDiv']
FirstOrderIndexes = {}
for i,metric in enumerate(Metrics):
    FirstOrderIndexes[metric] = get_Sensitivity_indexes(i,InputsFirstOrder) 


Vars = ['Burnup','Ctime','Nratios','Uncertainty','PriorType']
Indices = [str(i)+str(j) for i in range(len(Vars)) for j in range(i+1,len(Vars))]
Metrics = ['Distance','RelErrorBurnup','RelErrorCtime','KlDiv']

SumFirstAndSecond = {}
for metric in Metrics:
    SumFirstAndSecond[metric] = {}
    for i in range(len(Vars)):
        SumFirstAndSecond[metric][Vars[i]] = FirstOrderIndexes[metric][i] +\
           sum([SecondOrderIndexes[metric][Indices.index(j)] for j in Indices if str(i) in j])
           
           





    
    


