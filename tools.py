

import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.optimize as opt
import math
from datetime import date


def rightdays(df,time_shift,datesIS,datesOS):
    df['new_col'] = df['Timestringnum'].apply(lambda x: str(int(x))[0:])
    df['new_col'] = df['new_col'].astype(int)

    IS = df[(df['new_col']>= date.toordinal(datesIS[0])-time_shift)&(df.new_col <= date.toordinal(datesIS[1])-time_shift)]
    IS = IS.drop('new_col', axis=1)
    
    OS = df[(df['new_col']>= date.toordinal(datesOS[0])-time_shift)&(df.new_col <= date.toordinal(datesOS[1])-time_shift)]
    OS = OS.drop('new_col', axis=1)
    return [IS,OS]


def righthoursIS(df,hours):
    df['new_col'] = df['Timestringnum'].apply(lambda x: str(x-int(x))[1:])
    df['new_col'] = df['new_col'].astype(float)
    dff = df[(df['new_col']>=(hours[0]/24))&(df['new_col']<= (hours[1]/24))]
    dff = dff.drop('new_col', axis=1) 
    return dff


def righthoursOS(df,hours): 
    df['new_col'] = df['Timestringnum'].apply(lambda x: str(x-int(x))[1:])
    df['new_col'] = df['new_col'].astype(float)
    dff = df[(df['new_col']<=(hours[0]/24+0.00001))|(df['new_col'] >= (hours[1]/24-0.00001))]
    dff = dff.drop('new_col', axis=1)  
    return dff


def logratio(df,cHO,cLGO):
    logratio = []
    for z in zip(df['BidHOc2'],df['BidLGOc6'],df['AskHOc2'],df['AskLGOc6']):
        logratio.append(math.log((z[0] + z[2])*cHO / (z[1] + z[3])*cLGO))
    return logratio


def outliercheck(df):
    Q1 = np.percentile(df['logratio'],25)
    Q3 = np.percentile(df['logratio'],75)
    IQR = Q3 - Q1
    Outdf = df[(((df['logratio']-df['logratio'].shift(+1))>IQR) & ((df['logratio']-df['logratio'].shift(-1))>IQR*0.95))
                |((df['logratio']-df['logratio'].shift(-1))>IQR) & ((df['logratio']-df['logratio'].shift(+1))>IQR*0.95)]
    df_OC = pd.concat([df,Outdf,Outdf]).drop_duplicates(keep=False)
    return [df_OC,Outdf]


def MLE_estimator(logratio, dt):
    N = logratio.shape[0]
    X_m = logratio.iloc[:-1]
    X_p = logratio.iloc[1:]
    X_0 = logratio.iloc[0]
    X_N = logratio.iloc[-1]
    Y_m = X_m.mean(axis=0)
    Y_p = X_p.mean(axis=0)
    Y_mm = (X_m ** 2).mean(axis=0)
    Y_mp = (logratio.shift(-1) * logratio).mean(axis=0)
    Y_pp = (X_p ** 2).mean(axis=0)
    k = -np.log((Y_mp - Y_m * Y_p) / (Y_mm - Y_m ** 2)) / dt
    eta = Y_p + (X_N - X_0) / N * (Y_mp - Y_m * Y_p) / (Y_mm - Y_m ** 2 - (Y_mp - Y_m * Y_p))
    zeta_sq = Y_pp - Y_p ** 2 - (Y_mp - Y_m * Y_p) ** 2 / (Y_mm - Y_m ** 2)
    sigma = np.sqrt(2 * k * zeta_sq / (1 - np.exp(-2 * k * dt)))
    return [k, eta, sigma]


def costs(df):
    cost = ((df['AskHOc2']/df['BidHOc2']).apply(lambda x: str(np.log(x))[0:]).astype(float)
           +(df['AskLGOc6']/df['BidLGOc6']).apply(lambda x: str(np.log(x))[0:]).astype(float)).mean(axis=0)
    return cost


def long_run(loss, cost, theta, SIGMA, leverage, c):
    if leverage == -1:
        band = opt.fmin(mu_opt, [1, 0], args=(loss, cost, theta, SIGMA, c))
        optimallev = f_opt(band[0], band[1], loss, cost, theta, SIGMA, c)
        return [band,optimallev]
    else:
        band = opt.fmin(mu, [1, 0], args=(loss, cost, theta, SIGMA, leverage, c))
        return [band,leverage]
    
#Functions section

def erfi(x):
    return integrate.quad(lambda t: 2 / np.sqrt(np.pi) * np.exp(t ** 2), 0, x)[0]


def phi(x):
    return np.sqrt(np.pi) / 2 * erfi(x)


def erfid(x, y):
    return erfi(x / np.sqrt(2)) - erfi(y / np.sqrt(2))


def p_plus(u, d, loss):
    return (phi(d / np.sqrt(2)) - phi(loss / np.sqrt(2))) / (phi(u / np.sqrt(2)) - phi(loss / np.sqrt(2)))


def p_minus(u, d, loss):
    return (phi(u / np.sqrt(2)) - phi(d / np.sqrt(2))) / (phi(u / np.sqrt(2)) - phi(loss / np.sqrt(2)))


def v_plus(u, d, SIGMA, c):
    return np.exp((u - d - c) * SIGMA) - 1


def v_minus(d, loss, SIGMA, c):
    return np.exp((loss - d - c) * SIGMA) - 1


def q_plus(u, d, loss, SIGMA, c):
    return v_minus(d, loss, SIGMA, c) / (v_minus(d, loss, SIGMA, c) - v_plus(u, d, SIGMA, c))


def f_opt(u, d, loss, cost, theta, SIGMA, c):
    return -(p_plus(u, d, loss) / v_minus(d, loss, SIGMA, c) + p_minus(u, d, loss) / v_plus(u, d, SIGMA, c)) * (p_plus(u, d, loss) > q_plus(u, d, loss, SIGMA, c))


def mu_long_opt(x, loss, cost, theta, SIGMA, c):
    return (np.log(1 + f_opt(x[0], x[1], loss, cost, theta, SIGMA, c) * v_plus(x[0], x[1], SIGMA, c)) / erfid(x[0], x[1]) + np.log(1 + f_opt(x[0], x[1], loss, cost, theta, SIGMA, c) * v_minus(x[1], loss, SIGMA, c)) / erfid(x[1], loss)) / (np.pi * theta)


def mu_opt(x, loss, cost, theta, SIGMA, c):
    return -2 * float(mu_long_opt(x, loss, cost, theta, SIGMA, c)) + 10000000000.0 * int(x[0] - x[1] < c) + 10000000000.0 * int(x[1] - loss < 0)


def mu_long(x, loss, cost, theta, SIGMA, leverage, c):
    return (np.log(1 + leverage * v_plus(x[0], x[1], SIGMA, c)) / erfid(x[0], x[1]) + np.log(1 + leverage * v_minus(x[1], loss, SIGMA, c)) / erfid(x[1], loss)) / (np.pi * theta)


def mu(x, loss, cost, theta, SIGMA, leverage, c):
    return -2 * float(mu_long(x, loss, cost, theta, SIGMA, leverage, c)) + 10000000000.0 * int(x[0] - x[1] < c) + 10000000000.0 * int(x[1] - loss < 0)

###

def generateOU(k,eta,sigma,x0,dt,N_step):
    t_s = np.arange(0,dt*(N_step+1),dt)
    time_serie = [x0]
    stnormal = np.random.normal(0,1,N_step-1)
    for i in range(N_step-1):
        one = eta + (time_serie[i]-eta)*np.exp(-k*dt) + sigma*np.exp(-k*t_s[i+1])*np.sqrt((np.exp(2*k*t_s[i+1])-np.exp(2*k*t_s[i]))/(2*k))*stnormal[i]
        time_serie.append(one)
    return time_serie


def statisticalbootstrap(k,eta,sigma,dt,N_sample,N_steps,x0,leverage,loss,cost,c):
    parameters = []
    bands = [[],[],[],[]]
    for i in range(int(N_sample)):
        OUsim = generateOU(k,eta,sigma,x0,dt,N_steps)
        [ks, etas, sigmas] = MLE_estimator(pd.Series((v for v in OUsim)),dt)
        parameters.append([ks, etas, sigmas])
        theta = 1/ks
        SIGMA = sigmas/np.sqrt(2*ks);
        for j in range(len(leverage)):
            bandss = long_run(loss,cost,theta,SIGMA,leverage[j],c)
            bands[j].append(bandss)
    return  parameters,bands

        
def tradingStrategy(U, D, L, leverage, W0, time_strategy, OSS_OC, cost, eta):
    long = 0
    short = 0
    Wt = W0
    check_in = []
    check_out = []
    count = -1
    if OSS_OC['logratio'].iloc[0] > float(-D + eta):
        exitloc = 1
    elif (OSS_OC['logratio'].iloc[0] < float(-D + eta)) & (OSS_OC['logratio'].iloc[0] > float(D + eta)):
        exitloc = 2
    else:
        exitloc = 3
    for item in OSS_OC['logratio']:
        count += 1
        if long & (item > float(L + eta)) & (item < float(U + eta)):
            continue
        if short & (item > float(-U + eta)) & (item < float(-L + eta)):
            continue
        if long & (item >= float(U + eta)):
            check_out.append([count, item])
            Wt = Wt * (1 + leverage * (np.exp(item - openingPos - cost) - 1))
            long = 0
            exitloc = 2
            continue
        if short & (item <= float(-U + eta)):
            check_out.append([count, item])
            Wt = Wt * (1 + leverage * (np.exp(-item + openingPos - cost) - 1))
            short = 0
            exitloc = 2
            continue
        if long & (item <= float(L + eta)):
            check_out.append([count, item])
            Wt = Wt * (1 + leverage * (np.exp(item - openingPos - cost) - 1))
            long = 0
            exitloc = 3
            continue
        if short & (item >= float(-L + eta)):
            check_out.append([count, item])
            Wt = Wt * (1 + leverage * (np.exp(item - openingPos - cost) - 1))
            short = 0
            exitloc = 1
            continue
        #Se arriva fino a qua vuol dire che la posizione al momento è chiusa
        if exitloc == 1:
            if item <= float(-D + eta):
                check_in.append([count, item])
                openingPos = item
                short = 1
                continue
        if exitloc == 2:
            if item >= float(-D + eta):
                check_in.append([count, item])
                openingPos = item
                short = 1
                continue
            if item <= float(D + eta):
                check_in.append([count, item])
                openingPos = item
                long = 1
                continue
        if exitloc == 3:
            if item >= float(D + eta):
                check_in.append([count, item])
                openingPos = item
                long = 1
                continue
    if long:
        Wt = Wt * (1 + leverage * (np.exp(OSS_OC['logratio'].iloc[(-1)] - openingPos - cost) - 1))
        check_out.append([count, item])
    elif short:
            Wt = Wt * (1 + leverage * (np.exp(-OSS_OC['logratio'].iloc[(-1)] + openingPos - cost) - 1))
            check_out.append([count, item])
    log_return = 1 / time_strategy * np.log(Wt / W0)
    return (log_return, Wt, check_in, check_out)
