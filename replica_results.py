import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import re

ReHerrs = []
ReEerrs = []
ReHTerrs = []

ReHvars = []
ReEvars = []
ReHTvars = []

f = open("replica_results.out.txt", "r")
for x in f:
    if x[5:9] == 'ReH ':
        ReHerrs.append(list(map(float, re.findall('\d+\.\d+', x)))[1])
        ReHvars.append(list(map(float, re.findall('\d+\.\d+', x)))[2])
    if x[5:9] == 'ReE ':
        ReEerrs.append(list(map(float, re.findall('\d+\.\d+', x)))[1])
        ReEvars.append(list(map(float, re.findall('\d+\.\d+', x)))[2])
    if x[5:10] == 'ReHT ':
        ReHTerrs.append(list(map(float, re.findall('\d+\.\d+', x)))[1])
        ReHTvars.append(list(map(float, re.findall('\d+\.\d+', x)))[2])

avgErrReH = sum(ReHerrs)/len(ReHerrs)
avgErrReE = sum(ReEerrs)/len(ReEerrs)
avgErrReHT = sum(ReHTerrs)/len(ReHTerrs)

avgVarReH = sum(ReHvars)/len(ReHvars)
avgVarReE = sum(ReEvars)/len(ReEvars)
avgVarReHT = sum(ReHTvars)/len(ReHTvars)

print('Avg Err ReH = %.2f' % (avgErrReH))
print('Avg variance ReH = %.2f' % (avgVarReH))
print('Avg Err ReH = %.2f' % (avgErrReH))
print('Avg variance ReE = %.2f' % (avgVarReE))
print('Avg Err ReH = %.2f' % (avgErrReH))
print('Avg variance ReHT = %.2f' % (avgVarReHT))

p1 = plt.figure()
ax1 = p1.add_subplot(111)
ax1.plot(np.linspace(0, len(ReHerrs), len(ReHerrs)), ReHerrs, 'ro', label='Error in ReH')
ax1.plot(np.linspace(0, len(ReHerrs), len(ReHerrs)), np.zeros(len(ReHerrs))+avgErrReH, 'r--', label='Average Error in ReH')
ax1.errorbar(x = np.linspace(0, len(ReHerrs), len(ReHerrs)), y = ReHerrs, yerr = ReHvars, fmt = '.k')

ax2 = p1.add_subplot(111)
ax2.plot(np.linspace(0, len(ReEerrs), len(ReEerrs)), ReEerrs, 'go', label='Error in ReE')
ax2.plot(np.linspace(0, len(ReEerrs), len(ReEerrs)), np.zeros(len(ReEerrs))+avgErrReE, 'g--', label='Average Error in ReE')
ax2.errorbar(x = np.linspace(0, len(ReEerrs), len(ReEerrs)), y = ReEerrs, yerr = ReEvars, fmt = '.k')

ax3 = p1.add_subplot(111)
ax3.plot(np.linspace(0, len(ReHTerrs), len(ReHTerrs)), ReHTerrs, 'bo', label='Error in ReHT')
ax3.plot(np.linspace(0, len(ReHTerrs), len(ReHTerrs)), np.zeros(len(ReHTerrs))+avgErrReHT, 'b--', label='Average Error in ReHT')
ax3.errorbar(x = np.linspace(0, len(ReHTerrs), len(ReHTerrs)), y = ReHTerrs, yerr = ReHTvars, fmt = '.k')

plt.legend()
plt.show()