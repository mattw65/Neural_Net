import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import re
import sys

ReHs = []
ReEs = []
ReHTs = []

original_stdout = sys.stdout

with open('all_reps.txt', 'r') as f:
    for line in f:
        # ReHs.append(int(f[l][1]))
        # ReEs.append(int(f[4]))
        # ReHTs.append(int(line[6]))
        if len(line.split()) > 1:
            ReHs.append(float(line.split()[1]))
            ReEs.append(float(line.split()[2]))
            ReHTs.append(float(line.split()[3]))

ReHs = np.asarray(ReHs)
ReEs = np.asarray(ReEs)
ReHTs = np.asarray(ReHTs)

meanReH = np.mean(ReHs)
varReH = np.var(ReHs)
meanReE = np.mean(ReEs)
varReE = np.var(ReEs)
meanReHT = np.mean(ReHTs)
varReHT = np.var(ReHTs)

plt.hist(ReHs, bins=100, range = (meanReH - 3*np.sqrt(varReH), meanReH + 3*np.sqrt(varReH)))
plt.hist(ReEs, bins=100, range = (meanReE - 3*np.sqrt(varReE), meanReE + 3*np.sqrt(varReE)))
plt.show()



