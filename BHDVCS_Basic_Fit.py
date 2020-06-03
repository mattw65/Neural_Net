import csv
import numpy as np
import BHDVCS as BHDVCS
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lmfit import Model, Parameters, fit_report, minimize

b = BHDVCS.BHDVCS()

sumReH = []
sumReE = []
sumReHT = []
chisq = []

#Choose which method is used to fit data
#Options: leastsq, least_squares, differential_evolution, brute, basinhopping, etc. (more found at https://lmfit.github.io/lmfit-py/fitting.html)
fit_method = 'leastsq'

with open('DVCS_cross_fixed_t.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    xdata = []
    ydata = []
    expect = []

    rownum = 0

    #Read in data from file
    for row in readCSV:
        if rownum !=0:
            
            ind = float(row[0])         # index 
            k = float(row[1])           # k (Energy of incoming electron)
            qq = float(row[2])          # Q^2 (Electron squared momentum transfer)
            x_b = float(row[3])         # X_b (Bjorken Variable)
            t = float(row[4])           # t (Squared momentum transfer to the proton)
            phi_x = float(row[5])       # phi_x
            F = float(row[6])           # F (Cross section readings)
            errF = float(row[7])        # errF (Gaussian error in F)
            F1 = float(row[8])          # F1 (Elastic form factor 1)
            F2 = float(row[9])          # F12 (Elastic form factor 1)
            ReH_expc = float(row[10])   # Actual value for ReH
            ReE_expc = float(row[11])   # Actual value for ReE
            ReHT_expc = float(row[12])  # Actual value for ReHT
            
            #Add Phi_x values to list for x axis data in cross section
            xdata.append(phi_x)
            #Add F values for y axis values for data fitting
            ydata.append(F)

            #Get data from each set of index 0-35 and find best fit values for parameters
            if ind == 35:
                #turn data lists into numpy arrays for curve fitting
                xdat  = np.array(xdata)
                ydat = np.array(ydata)

                #Set parameters for ReH ReE and ReHT, all bounded between 1 and 0
                fit_params = Parameters()
                fit_params.add('ReH', value = 0.5, min = 0, max = 1)
                fit_params.add('ReE', value = 0.5, min = 0, max = 1)
                fit_params.add('ReHT', value = 0.5, min = 0, max = 1)

                #Fit function modified from BHDVCS.py
                def func(phi, ReH, ReE, ReHT):
                    const = 0.014863
                    b.SetKinematics(qq, x_b, t, k)
                    b.Set4VectorsPhiDep(phi)
                    b.Set4VectorProducts(phi)
                    xsbhuu = b.GetBHUUxs(F1, F2)
                    xsiuu = b.GetIUUxs(phi, F1, F2, ReH, ReE, ReHT)
                    return xsbhuu + xsiuu + const

                #Create model using fit function
                BHDVCSmodel = Model(func)
                #Fit model using data and parameters
                result = BHDVCSmodel.fit(ydat, fit_params, phi = xdat, method = fit_method)
                
                #Add best fit values for ReH, ReE, and ReHT in current data set to list
                sumReH.append(result.best_values['ReH'])
                sumReE.append(result.best_values['ReE'])
                sumReHT.append(result.best_values['ReHT'])

                #Add chi squared value of each fit to list
                chisq.append(result.chisqr)
                
                #Clear data lists for next trial
                del xdata[:]
                del ydata[:]                
                        
        rownum = rownum + 1

#Calculate averages across data
avg_ReH = sum(sumReH)/len(sumReH)
avg_ReE = sum(sumReE)/len(sumReE)
avg_ReHT = sum(sumReHT)/len(sumReHT)

#Calculate Errors
err_ReH = 100*(abs(avg_ReH-ReH_expc))/ReH_expc
err_ReE = 100*(abs(avg_ReE-ReE_expc))/ReE_expc
err_ReHT = 100*(abs(avg_ReHT-ReHT_expc))/ReHT_expc

print('%25s%.2f' % ('Avg Fit Value of ReH = ', avg_ReH))
print('%25s%.2f' % ('Actual Value of ReH = ', ReH_expc))
print('%25s%.2f\n' % ('Percent Error (ReH) = ', err_ReH))

print('%25s%.2f' % ('Avg Fit Value of ReE = ', avg_ReE))
print('%25s%.2f' % ('Actual Value of ReE = ', ReE_expc))
print('%25s%.2f\n' % ('Percent Error (ReE) = ', err_ReE))

print('%25s%.2f' % ('Avg Fit Value of ReHT = ', avg_ReHT))
print('%25s%.2f' % ('Actual Value of ReHT = ', ReHT_expc))
print('%25s%.2f\n' % ('Percent Error (ReHT) = ', err_ReHT))

print('%25s%.2f\n' % ('Avg Chi Sq of fit = ', sum(chisq)/len(chisq)))












