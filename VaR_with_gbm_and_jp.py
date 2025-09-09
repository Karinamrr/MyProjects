'''
Created by Karina Mendoza

Monte Carlo Simulation of Value-at-Risk under GBM and Jump-Diffusion Models

In this project we developed a Python program to estimate and compare Value-at-Risk (VaR) for selected assets
using Monte Carlo simulations under two stochastic processes: the Geometric Brownian Motion (GBM) and 
theMerton Jump-Diffusion (JD) model. The project includes historical calibration of drift, 
volatility, and jump parameters, simulation of price paths, distribution analysis of returns, and 
visualization of risk profiles

'''


import numpy as np
from numpy.random import standard_normal, poisson
import matplotlib.pyplot as plt 
import scipy.stats as sp
import pandas as pd



#Prepare data
data = pd.read_csv('https://hilpisch.com/tr_eikon_eod_data.csv', index_col=0, parse_dates=True)
assets=['AAPL.O', 'MSFT.O', 'AMZN.O']


T=30/365.
simulations=10000
percs=[0.01, 0.1, 1., 2.5, 5.0, 10.0]

def gbm(asset):
    S=data[asset]
    S0=data[asset].iloc[-1]
    returns = np.log(S/S.shift(1)).dropna()

    #GBM parameters
    r=returns.mean()*252
    sigma=returns.std()*np.sqrt(252)


    #Using the gbm formula 
    ST=S0*np.exp((r-0.5*sigma**2)*T+sigma*np.sqrt(T)*standard_normal(simulations))

    #Calculate and sort abs returns
    R_gbm = np.sort(ST - S0)


    plt.figure(figsize=(10, 6))
    plt.hist(R_gbm, bins=50)
    plt.xlabel('absolute return')
    plt.ylabel('frequency')
    plt.title(f'{asset}')
    plt.show()
    
    var=sp.scoreatpercentile(R_gbm, percs)


    print('%16s %16s' % ('Confidence Level', 'Value-at-Risk'))
    print(33 * '-')
    for pair in zip(percs, var):
        print('%16.2f %16.3f' % (100 - pair[0], -pair[1]))
    return R_gbm


def jd(asset):
    S=data[asset]
    S0=data[asset].iloc[-1]
    returns = np.log(S/S.shift(1)).dropna()
    sigma=returns.std()*np.sqrt(252)
    r=returns.mean()*252
    
    #Jumps detection
    thereshold=3*returns.std()
    jumps=returns[np.abs(returns)>thereshold]

    #Jump parameters
    lambda_jumps= len(jumps)/(len(returns)/252)
    mu=jumps.mean()
    delta=jumps.std()

    #Drift adjustment
    r_j = lambda_jumps*(np.exp(mu+0.5* delta**2)-1)
    r_total=r-r_j

    #Discretization
    M=100
    dt=T/M

    #Arrays simulation
    S=np.zeros((M+1, simulations))
    S[0]=S0

    #Random numbers
    sn1=standard_normal((M+1, simulations))
    sn2=standard_normal((M+1, simulations))
    poi=poisson(lambda_jumps*dt,(M+1, simulations))

    for t in range(1, M+1, 1):
        S[t]=S[t-1]*(np.exp((r_total-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*sn1[t])+ (np.exp(mu+delta*sn2[t])-1)*poi[t])
        S[t] = np.maximum(S[t], 0)

    R_jd=np.sort(S[-1]-S0)
    plt.figure(figsize=(10, 6))
    plt.hist(R_jd, bins=50)
    plt.xlabel('absolute return')
    plt.ylabel('frequency');
    plt.title(f'{asset}')
    plt.show()


    var2 = sp.scoreatpercentile(R_jd, percs)
    print('%16s %16s' % ('Confidence Level', 'Value-at-Risk'))
    print(33 * '-')
    for pair in zip(percs, var2):
        print('%16.2f %16.3f' % (100 - pair[0], -pair[1]))
    return R_jd

def comparisson_plot(R_gbm,R_jd):
    percs = list(np.arange(0.0, 10.1, 0.1))
    gbm_var = sp.scoreatpercentile(R_gbm, percs)
    jd_var = sp.scoreatpercentile(R_jd, percs)

    plt.figure(figsize=(10, 6))
    plt.plot(percs, gbm_var, 'b', lw=1.5, label='GBM')
    plt.plot(percs, jd_var, 'r', lw=1.5, label='JD')
    plt.legend(loc=4)
    plt.xlabel('100 - confidence level [%]')
    plt.ylabel('value-at-risk')
    plt.ylim(ymax=0.0)
    plt.plot()


for i in assets:
    a= gbm(i)
    b=jd(i)
    comparisson_plot(a,b)