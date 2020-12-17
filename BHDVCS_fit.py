import csv
import math
import numpy as np
import pandas as pd
from BHDVCS import BHDVCS
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.stats import chisquare
from statistics import median
import lmfit

b = BHDVCS()
f = b.TotalUUXS_curve_fit
g = b.TotalUUXS_curve_fit2

dats = pd.read_csv('dvcs_psuedo.csv')
dats2 = pd.read_csv('DVCS_cross_fixed_t.csv')

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
ReH_exp = np.array(dats['ReH'])
ReE_exp = np.array(dats['ReE'])
ReHT_exp = np.array(dats['ReHtilde'])

def fit_scipy(n):
    
    a = (np.amax(ind)+1)*n
    b = a + np.amax(ind)

    xdat = (phi[a:b], qq[a:b], xb[a:b], t[a:b], k[a:b], F1[a:b], F2[a:b], const[a:b])
    popt, pcov = curve_fit(f, xdat, ydat[a:b], sigma=errF[a:b], method='lm')

    #Calculate Errors
    err_H = abs(100*(abs(popt[0]-ReH_exp[a]))/ReH_exp[a])
    err_E = abs(100*(abs(popt[1]-ReE_exp[a]))/ReE_exp[a])
    err_HT = abs(100*(abs(popt[2]-ReHT_exp[a]))/ReHT_exp[a])

    # print('\n\n%25s%.2f' % ('Fit Value of ReH = ', popt[0]))
    # print('%25s%.2f' % ('Actual Value of ReH = ', ReH_exp[a]))
    # print('%25s%.1f%%' % ('Error (ReH) = ', err_H))

    # print('%25s%.2f' % ('Fit Value of ReE = ', popt[1]))
    # print('%25s%.2f' % ('Actual Value of ReE = ', ReE_exp[a]))
    # print('%25s%.1f%%' % ('Error (ReE) = ', err_E))

    # print('%25s%.2f' % ('Fit Value of ReHT = ', popt[2]))
    # print('%25s%.2f' % ('Actual Value of ReHT = ', ReHT_exp[a]))
    # print('%25s%.1f%%' % ('Error (ReHT) = ', err_HT))
    #print('Average Error for set #%d using scipy = %.2f%%' % (n, (abs(err_H)+abs(err_E)+abs(err_HT))/3))


    # plt.plot(phi[a:b], ydat[a:b], 'bo', label='Given F Values')
    # plt.plot(phi[a:b], f(xdat, *popt), 'g--', label='Line of Best Fit: ReH=%5.3f, ReE=%5.3f, ReHT=%5.3f' % tuple(popt))
    # plt.legend()
    # plt.show()

    return err_H, err_E, err_HT

def fit_lm(n):
    
    a = (np.amax(ind)+1)*n
    b = a + np.amax(ind)

    phis = phi[a:b]
    xdat = (qq[a:b], xb[a:b], t[a:b], k[a:b], F1[a:b], F2[a:b], const[a:b])
    
    dvcs = lmfit.Model(g)
   
    params = lmfit.Parameters()
    params.add('qq', value = qq[a], vary=False)
    params.add('xb', value = xb[a], vary=False)
    params.add('t', value = t[a], vary=False)
    params.add('k', value = k[a], vary=False)
    params.add('F1', value = F1[a], vary=False)
    params.add('F2', value = F2[a], vary=False)
    params.add('const', value = const[a], vary=False)
    params.add('ReH', value = 1, min = -100, max = 100)
    params.add('ReE', value = 1, min = -100, max = 100)
    params.add('ReHT', value = 1, min = -100, max = 100)

    # params.add('ReH', value = 1)
    # params.add('ReE', value = 1)
    # params.add('ReHT', value = 1)

    result = dvcs.fit(ydat[a:b], params, phi = phis, method = 'leastsq')
    # print(result.fit_report())

    ReHfit = result.best_values['ReH']
    ReEfit = result.best_values['ReE']
    ReHTfit = result.best_values['ReHT']

    #Calculate Errors
    err_H = (100*(abs(ReHfit-ReH_exp[a]))/ReH_exp[a])
    err_E = (100*(abs(ReEfit-ReE_exp[a]))/ReE_exp[a])
    err_HT = (100*(abs(ReHTfit-ReHT_exp[a]))/ReHT_exp[a])

    # print('%25s%.2f' % ('Fit Value of ReH = ', ReHfit))
    # print('%25s%.2f' % ('Actual Value of ReH = ', ReH_exp[a]))
    # print('%25s%.1f%%\n' % ('Error (ReH) = ', err_H))

    # print('%25s%.2f' % ('Fit Value of ReE = ', ReEfit))
    # print('%25s%.2f' % ('Actual Value of ReE = ', ReE_exp[a]))
    # print('%25s%.1f%%\n' % ('Error (ReE) = ', err_E))

    # print('%25s%.2f' % ('Fit Value of ReHT = ', ReHTfit))
    # print('%25s%.2f' % ('Actual Value of ReHT = ', ReHT_exp[a]))
    # print('%25s%.1f%%\n' % ('Error (ReHT) = ', err_HT))

    return err_H, err_E, err_HT

def avgs_scipy1():
    
    err_H = []
    err_E = []
    err_HT = []

    for r in range(15):
        print(r)
        errs = fit_scipy(r)
        err_H.append(errs[0])
        err_E.append(errs[1])
        err_HT.append(errs[2])

    avgerrH = sum(err_H)/len(err_H)
    avgerrE = sum(err_E)/len(err_E)
    avgerrHT = sum(err_HT)/len(err_HT)

    print('%25s%.1f%%' % ('Average Error scipy (ReH) = ', avgerrH))
    print('%25s%.1f%%' % ('Average Error scipy (ReE) = ', avgerrE))
    print('%25s%.1f%%' % ('Average Error scipy (ReHT) = ', avgerrHT))
    
def avgs_LM():
    
    err_H = []
    err_E = []
    err_HT = []

    for set in range(np.amax(n)):
        errs = fit_lm(set)
        err_H.append(errs[0])
        err_E.append(errs[1])
        err_HT.append(errs[2])

    avgerrH = sum(err_H)/len(err_H)
    avgerrE = sum(err_E)/len(err_E)
    avgerrHT = sum(err_HT)/len(err_HT)

    print('%25s%.1f%%' % ('Average Error lm (ReH) = ', avgerrH))
    print('%25s%.1f%%' % ('Average Error lm (ReE) = ', avgerrE))
    print('%25s%.1f%%\n' % ('Average Error lm (ReHT) = ', avgerrHT))			


avgs_scipy1()
#avgs_LM()
#fit_scipy(0)
# fit_lm(1)

