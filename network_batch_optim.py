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

from scipy.stats import chisquare

import time
import sys

tb = TBHDVCS()

f = tb.TotalUUXS_curve_fit3
loss_func = tb.loss_MSE

datset = int(sys.argv[1])

dats = pd.read_csv('dvcs_psuedo.csv')
n = np.array(dats['#Set'])
ind = np.array(dats['index'])
k = np.array(dats['k'])
qq = np.array(dats['QQ'])
xb = np.array(dats['x_b'])
t = np.array(dats['t'])
phi = np.array(dats['phi_x'])
ydat = np.array(dats['F'])
errF = np.array(dats['errF']) 
F1 = np.array(dats['F1'])
F2 = np.array(dats['F2'])
const = np.array(dats['dvcs'])  
ReH_target = np.array(dats['ReH']) 
ReE_target = np.array(dats['ReE']) 
ReHT_target = np.array(dats['ReHtilde'])

node_ops = [100, 75, 50, 25, 10]
learning_rate_ops = [0.005, 0.01, 0.05, 0.1, 0.5]

best = 100

a = datset*36
b = a + 36


xdat = np.asarray([phi[a:b], qq[a:b], xb[a:b], t[a:b], k[a:b], F1[a:b], F2[a:b], const[a:b]])
x = Variable(torch.from_numpy(xdat[1:5].transpose()))
y = Variable(torch.from_numpy(ydat[a:b].transpose()))
xdat = Variable(torch.from_numpy(xdat))
errs = Variable(torch.from_numpy(errF[a:b]))


start = time.time()
# step = 0

for nodes in node_ops:
    layers=0
    net_ops = [
    torch.nn.Sequential(
        torch.nn.Linear(4, nodes),
        torch.nn.Tanh(),
        torch.nn.Linear(nodes, nodes),
        torch.nn.Tanh(),
        torch.nn.Linear(nodes, nodes),
        torch.nn.Tanh(),
        torch.nn.Linear(nodes, nodes),
        torch.nn.Tanh(),
        torch.nn.Linear(nodes, nodes),
        torch.nn.Tanh(),
        torch.nn.Linear(nodes, nodes),
        torch.nn.Tanh(),
        torch.nn.Linear(nodes, nodes),
        torch.nn.Tanh(),
        torch.nn.Linear(nodes, nodes),
        torch.nn.Tanh(),
        torch.nn.Linear(nodes, nodes),
        torch.nn.Tanh(),
        torch.nn.Linear(nodes, nodes),
        torch.nn.Tanh(),
        torch.nn.Linear(nodes, nodes),
        torch.nn.Tanh(),
        torch.nn.Linear(nodes, 3)
    ),

    torch.nn.Sequential(
        torch.nn.Linear(4, nodes),
        torch.nn.Tanh(),
        torch.nn.Linear(nodes, nodes),
        torch.nn.Tanh(),
        torch.nn.Linear(nodes, nodes),
        torch.nn.Tanh(),
        torch.nn.Linear(nodes, nodes),
        torch.nn.Tanh(),
        torch.nn.Linear(nodes, nodes),
        torch.nn.Tanh(),
        torch.nn.Linear(nodes, nodes),
        torch.nn.Tanh(),
        torch.nn.Linear(nodes, 3)
    ),

    torch.nn.Sequential(
        torch.nn.Linear(4, nodes),
        torch.nn.Tanh(),
        torch.nn.Linear(nodes, nodes),
        torch.nn.Tanh(),
        torch.nn.Linear(nodes, nodes),
        torch.nn.Tanh(),
        torch.nn.Linear(nodes, 3)
    )
    ]
    for net in net_ops:
        training_net = net
        layers+=1
        for l in learning_rate_ops:
                    # step+=1

                    optimizer = torch.optim.Adam(training_net.parameters(), lr=l)

                    for epoch in range(2500):
                        preds = training_net(x.float()) #output 3 predicted values for cffs

                        ReHfit = torch.mean(torch.transpose(preds, 0, 1)[0])
                        ReEfit = torch.mean(torch.transpose(preds, 0, 1)[1])
                        ReHTfit = torch.mean(torch.transpose(preds, 0, 1)[2])
                        cffs = [ReHfit, ReEfit, ReHTfit]

                        loss = loss_func((xdat.float()), cffs, errs, y)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    ReHfit = ReHfit.data.numpy()
                    ReEfit = ReEfit.data.numpy()
                    ReHTfit = ReHTfit.data.numpy()
                    fit_cffs = [ReHfit, ReEfit, ReHTfit]
                    
                    err_H = abs(100*((fit_cffs[0]-ReH_target[a]))/ReH_target[a])
                    err_E = abs(100*((fit_cffs[1]-ReE_target[a]))/ReE_target[a])
                    err_HT = abs(100*((fit_cffs[2]-ReHT_target[a]))/ReHT_target[a])
                    
                    avgErr = (err_H + err_E + err_HT)/3
                    
                    if avgErr < best:
                        best = avgErr
                        optims = [err_H, err_E, err_HT, layers, nodes, l]

                    # pctdone = (100*step/((len(learning_rate_ops))*(len(net_ops))*(len(node_ops))))
                    # sys.stdout.write('\r%.2f%% complete. %.1f minutes remaining.' % (pctdone, ((100/pctdone)*(time.time()-start)/60)-((time.time()-start)/60)))
                    # sys.stdout.flush()

                        
            
                                                  
print('\nSet #%d optimized to %.2f%%' % (datset, best))
print(optims[0])
print(optims[1])
print(optims[2])
print(optims[3])
print(optims[4])
print(optims[5])