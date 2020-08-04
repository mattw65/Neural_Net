import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import sys

import numpy as np
import pandas as pd 
from BHDVCS import BHDVCS

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import csv

class dvcsLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))
       
        
    def forward(self, input):
        x, y = input.shape
        if y != self.in_features:
            sys.exit(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
        
        print(x.shape)
        output = input @ self.weight.t() + self.bias
        return output


b = BHDVCS()
f = b.TotalUUXS_curve_fit

dats = pd.read_csv('dvcs_psuedo.csv')

n = np.array(dats['#Set'])[0:36]
ind = np.array(dats['index'])[0:36]
k = np.array(dats['k'])[0:36]
qq = np.array(dats['QQ'])[0:36]
xb = np.array(dats['x_b'])[0:36]
t = np.array(dats['t'])[0:36]
phi = np.array(dats['phi_x'])[0:36]
ydat = np.array(dats['F'])[0:36]  
errF = np.array(dats['errF'])[0:36]  
F1 = np.array(dats['F1'])[0:36]  
F2 = np.array(dats['F2'])[0:36]  
const = np.array(dats['dvcs'])[0:36]  
ReH_target = np.array(dats['ReH'])[0:36]  
ReE_target = np.array(dats['ReE'])[0:36]  
ReHT_target = np.array(dats['ReHtilde'])[0:36] 

xdat = np.asarray([phi, qq, xb, t, k, F1, F2, const])
xs = []
ys = []

with open('dvcs_psuedo.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        rownum = 11
        if row[0] != '#Set':
            rownum = int(row[0])
        if rownum < 10:
            xs.append(row[4:12])
            ys.append(row[3])


x = Variable(torch.from_numpy(np.asarray(xs).astype(float)))
y = Variable(torch.from_numpy(np.asarray(ys).astype(float)))

print(x.size())
print(y.size())


net = torch.nn.Sequential(
        torch.nn.Linear(36, 72),
        torch.nn.Tanh(),
        torch.nn.Linear(72, 36),
        torch.nn.Tanh(),
        torch.nn.Linear(36, 3)
    )


optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

EPOCH = 200

def loss_func(cffs, y_true):
  # y_true and y_pred are numpy arrays of the same length.
    y_pred = Variable(torch.from_numpy(f(xdat[:,0:360], cffs.data.numpy()[-1,0], cffs.data.numpy()[-1,1], cffs.data.numpy()[-1,2])), requires_grad = True)
    return (y_pred - y_true ** 2).mean()

# start training
for epoch in range(EPOCH):

    p = net(x.float()) # array 360 x 3

    loss = loss_func(p, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss)
    

xstest = []
ystest = []

with open('dvcs_psuedo.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        rownum = 13
        if row[0] != '#Set':
            rownum = int(row[0])
        if rownum == 12:
            xstest.append(row[4:12])
            ystest.append(row[3])

xtest = Variable(torch.from_numpy(np.asarray(xstest).astype(float)))
ytest = Variable(torch.from_numpy(np.asarray(ystest).astype(float)))
cffs = net(xtest.float())

predictions = f(xdat[:,(12*36):(13*36)], cffs.data.numpy()[-1,0], cffs.data.numpy()[-1,1], cffs.data.numpy()[-1,2])

plt.plot(phi[396:432], ydat[396:432], 'bo', label='data')
plt.plot(phi[396:432], predictions, 'g--', label='fit')
plt.legend()
plt.show()