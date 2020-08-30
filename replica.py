import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import numpy as np
import pandas as pd 
from BHDVCS_torch import TBHDVCS

import matplotlib
import matplotlib.pyplot as plt

import sys
from scipy.stats import chisquare

tb = TBHDVCS()

f = tb.TotalUUXS_curve_fit3
loss_func = tb.loss_MSE_errs

dats = pd.read_csv('dvcs_psuedo.csv')
n = np.array(dats['#Set'])
ind = np.array(dats['index'])
k = np.array(dats['k'])
qq = np.array(dats['QQ'])
xb = np.array(dats['x_b'])
t = np.array(dats['t'])
phi = np.array(dats['phi_x'])
F = np.array(dats['F'])
errF = np.array(dats['errF']) 
F1 = np.array(dats['F1'])
F2 = np.array(dats['F2'])
const = np.array(dats['dvcs'])  
ReH_target = np.array(dats['ReH']) 
ReE_target = np.array(dats['ReE']) 
ReHT_target = np.array(dats['ReHtilde'])
yrep = []

errs_H = []
errs_E = []
errs_HT = []

rep_ReH = []
rep_ReE = []
rep_ReHT = []

blank_net = torch.nn.Sequential(
        torch.nn.Linear(4, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 80),
        torch.nn.Tanh(),
        torch.nn.Linear(80, 3)
    )

optimizer = torch.optim.Adam(blank_net.parameters(), lr=0.02)

EPOCH = 2500

datset = int(sys.argv[1])

reps = 100

i = datset
a = 36*i # start index of set
b = a+36 # end index of set



for rep in range(reps): # create n replicas
    
    net = blank_net # untrain/reset network
    
    yrep = [0] * (b-a) # create array to be filled with replicated F values
    
    for l in range(b-a): # populate yrep with random normal values with mean = F and sd = errF
        
        yind = a+l # index of data point 
        yrep[l] = (np.random.normal(F[yind], errF[yind]))

    
    xdat = np.array([phi[a:b], qq[a:b], xb[a:b], t[a:b], k[a:b], F1[a:b], F2[a:b], const[a:b]])
    ydat = np.array(yrep)

    x = Variable(torch.from_numpy(xdat[1:5].transpose()))
    y = Variable(torch.from_numpy(ydat.transpose()))

    xdat = Variable(torch.from_numpy(xdat))

    errs = Variable(torch.from_numpy(errF[a:b]))

    for epoch in range(EPOCH):

        p = net(x.float()) #output 3 predicted values for cffs

        ReHfit = torch.mean(torch.transpose(p, 0, 1)[0])
        ReEfit = torch.mean(torch.transpose(p, 0, 1)[1])
        ReHTfit = torch.mean(torch.transpose(p, 0, 1)[2])
        cffs = [ReHfit, ReEfit, ReHTfit]

        loss = loss_func((xdat.float()), cffs, errs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    rep_ReH.append(cffs[0].data.numpy())
    rep_ReE.append(cffs[1].data.numpy())
    rep_ReHT.append(cffs[2].data.numpy())

rep_ReH = np.array(rep_ReH)
rep_ReE = np.array(rep_ReE)
rep_ReHT = np.array(rep_ReHT)

err_H = abs(100*(abs(np.mean(rep_ReH)-ReH_target[a]))/ReH_target[a])
err_E = abs(100*(abs(np.mean(rep_ReE)-ReE_target[a]))/ReE_target[a])
err_HT = abs(100*(abs(np.mean(rep_ReHT)-ReHT_target[a]))/ReHT_target[a])


print('\nMean ReH for set %d = %.2f, error = %.2f, variance = %.2f' % (i, np.mean(rep_ReH), err_H, np.var(rep_ReH)))
print('Mean ReE for set %d = %.2f, error = %.2f, variance = %.2f' % (i, np.mean(rep_ReE), err_E, np.var(rep_ReE)))
print('Mean ReHT for set %d = %.2f, error = %.2f, variance = %.2f\n' % (i, np.mean(rep_ReHT), err_HT, np.var(rep_ReHT)))