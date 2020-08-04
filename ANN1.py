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

xs = []
ys = []

with open('dvcs_psuedo.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if row[0] == '0':
            xs.append(row[4:12])
            ys.append(row[2])


x = np.asarray(xs).astype(float)
y = np.asarray(ys).astype(float)

x2 = np.asarray([phi, qq, xb, t, k, F1, F2, const])
y2 = ydat  

x, y = Variable(torch.from_numpy(x)), Variable(torch.from_numpy(y))

# plt.figure(figsize=(10,4))
# plt.scatter(phi, ydat, color = "blue")
# plt.title('Regression Analysis')
# plt.xlabel('Independent varible')
# plt.ylabel('Dependent varible')
# plt.savefig('curve_2.png')
# plt.show()

# another way to define a network
net = torch.nn.Sequential(
        torch.nn.Linear(8, 36),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(36, 36),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(36, 3),
    )

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

BATCH_SIZE = 14
EPOCH = 200

def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()

my_images = []
fig, ax = plt.subplots(figsize=(16,10))

# start training
for epoch in range(EPOCH):
    
    cff1 = []
    cff2 = []
    cff3 = []

    cffs = net(x.float()).data.numpy()

    for a in range(36):
        cff1.append(cffs[a][0])
        cff2.append(cffs[a][1])
        cff3.append(cffs[a][2])

    ReHs = np.asarray(cff1).astype(float)
    ReEs = np.asarray(cff2).astype(float)
    ReHTs = np.asarray(cff3).astype(float)

    prediction = f(x2, ReHs, ReEs, ReHTs)    # input x and predict based on x
    p = Variable(torch.from_numpy(prediction), requires_grad=True)
    loss = mse_loss(p, y)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients
    
    
    # plt.cla()
    # ax.set_title('Regression Analysis - model 3 Batches', fontsize=35)
    # ax.set_xlabel('Independent variable', fontsize=24)
    # ax.set_ylabel('Dependent variable', fontsize=24)
    # ax.set_xlim(-11.0, 13.0)
    # ax.set_ylim(-1.1, 1.2)
    # ax.scatter(phi, ydat, color = "blue", alpha=0.2)
    # ax.scatter(phi, p.data.numpy(), color='green', alpha=0.5)
    # ax.text(8.8, -0.8, 'Epoch = %d' % epoch,
    #         fontdict={'size': 24, 'color':  'red'})
    # ax.text(8.8, -0.95, 'Loss = %.4f' % loss.data.numpy(),
    #         fontdict={'size': 24, 'color':  'red'})

    # # Used to return the plot as an image array 
    # # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
    # fig.canvas.draw()       # draw the canvas, cache the renderer
    # image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    # image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # my_images.append(image)

    


# imageio.mimsave('./curve_2_model_3_batch.gif', my_images, fps=12)


ReHF = []
ReEF = []
ReHTF = []

cffs = net(x.float()).data.numpy()

for a in range(36):
    ReHF.append(cffs[a][0])
    ReEF.append(cffs[a][1])
    ReHTF.append(cffs[a][2])

ReHF = np.asarray(cff1).astype(float)
ReEF = np.asarray(cff2).astype(float)
ReHTF = np.asarray(cff3).astype(float)

prediction = f(x2, ReHF, ReEF, ReHTF)    # input x and predict based on x
# print(prediction)
p = Variable(torch.from_numpy(prediction), requires_grad=True)

plt.plot(phi[0:36], ydat[0:36], 'bo', label='data')
plt.plot(phi[0:36], p.data.numpy(), 'g--', label='fit')
plt.legend()
plt.show()