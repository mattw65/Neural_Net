import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

dats = pd.read_csv('dvcs_psuedo.csv')
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
ReH = np.array(dats['ReH']) 
ReE = np.array(dats['ReE']) 
ReHT = np.array(dats['ReHtilde'])

ReHs = []
ReEs = []
ReHTs = []
errReH = []
errReE = []
errReHT = []
ks = []
qqs = []
xbs = []
ts = []
layersReH = []
layersReE = []
layersReHT = []
lrReH = []
lrReE = []
lrReHT = []

for i in range(14):
    ReHs.append(ReH[36*i])
    ReEs.append(ReE[36*i])
    ReHTs.append(ReHT[36*i])
    ks.append(k[36*i])
    qqs.append(qq[36*i])
    xbs.append(xb[36*i])
    ts.append(t[36*i])
    filename = ('network_optim_' + str(i) + '.out.txt')
    optims = np.array(pd.read_csv(filename))
    errReH.append(optims[0][0])
    errReE.append(optims[1][0])
    errReHT.append(optims[2][0])
    layersReH.append(optims[3][0])
    layersReE.append(optims[4][0])
    layersReHT.append(optims[5][0])
    lrReH.append(optims[6][0])
    lrReE.append(optims[7][0])
    lrReHT.append(optims[7][0])

# index: 0     1     2     3   4    5    6     7       8      9         10          11         12        13     14      15
all3 = [ReHs, ReEs, ReHTs, ks, qqs, xbs, ts, errReH, errReE, errReHT, layersReH, layersReE, layersReHT, lrReH, lrReE, lrReHT]
names = ['ReH', 'ReE', 'ReHT', 'k', 'qq', "xb", 't', 'errReH', 'errReE', 'errReHT', 'layersReH', 'layersReE', 'layersReHT', 'lrReH', 'lrReE', 'lrReHT']
for a in range(len(all3)):
    print('---------------------------------------------')
    for b in range(len(all3)):
        if a != b and stats.pearsonr(all3[a], all3[b])[1] < 0.1:
            print('Correlation in %s vs %s: %.3f , %.3f' % (names[a], names[b], stats.pearsonr(all3[a], all3[b])[0], stats.pearsonr(all3[a], all3[b])[1]))
print((sum(errReH)+sum(errReE)+sum(errReHT))/(len(errReH)+len(errReE)+len(errReHT)))
# plt.plot(errReH, ks, 'bo', label='errReH vs k')
plt.plot(errReH, ks, 'ro', label='errReE vs k')
plt.legend()
plt.show()

ReHs1 = []
ReEs1 = []
ReHTs1 = []
errReH1 = []
errReE1 = []
errReHT1 = []
ks1 = []
qqs1 = []
xbs1 = []
ts1 = []
layers1 = []
nodes1 = []
lr1 = []

for i in range(14):
    ReHs1.append(ReH[36*i])
    ReEs1.append(ReE[36*i])
    ReHTs1.append(ReHT[36*i])
    ks1.append(k[36*i])
    qqs1.append(qq[36*i])
    xbs1.append(xb[36*i])
    ts1.append(t[36*i])
    filename = ('network1_optim_' + str(i) + '.out.txt')
    optims1 = np.array(pd.read_csv(filename))
    errReH1.append(optims1[0][0])
    errReE1.append(optims1[1][0])
    errReHT1.append(optims1[2][0])
    layers1.append(optims1[3][0])
    nodes1.append(optims1[4][0])
    lr1.append(optims1[5][0])
    

# index: 0     1     2     3   4    5    6     7       8      9         10          11         12        13     14      15
all1 = [ReHs1, ReEs1, ReHTs1, ks1, qqs1, xbs1, ts1, errReH1, errReE1, errReHT1, layers1, nodes1, lr1]
names = ['ReH', 'ReE', 'ReHT', 'k', 'qq', "xb", 't', 'errReH', 'errReE', 'errReHT', 'layers', 'nodes', 'learning rate']
for a in range(len(all1)):
    print('---------------------------------------------')
    for b in range(len(all1)):
        if a != b and stats.pearsonr(all1[a], all1[b])[1] < 0.1:
            print('Correlation in %s vs %s: %.3f , %.3f' % (names[a], names[b], stats.pearsonr(all1[a], all1[b])[0], stats.pearsonr(all1[a], all1[b])[1]))

print((sum(errReH1)+sum(errReE1)+sum(errReHT1))/(len(errReH1)+len(errReE1)+len(errReHT1)))
# plt.plot(errReH, ks, 'bo', label='errReH vs k')
# plt.plot(errReE1, errReH1, 'ro', label='errReE vs errReE')
# plt.legend()
# plt.show()