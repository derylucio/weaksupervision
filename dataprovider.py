from sklearn.preprocessing import StandardScaler
from utils import getjetvar
import numpy as np

def getSamples(features,etamax,bins):
    filename = 'data/20161106_19h36min.root'
    id0 = getjetvar('j0','id',filename)
    eta0 = getjetvar('j0','eta',filename)
    eta1 = getjetvar('j1','eta',filename)
    deta0 = getjetvar('j0','deta',filename)

    xx = [getjetvar('j0',var,filename) for var in features]
    X = np.vstack( [xx] ).T

    id0 = id0[(np.fabs(eta0)<etamax) & (np.fabs(eta1)<etamax)]
    deta0 = deta0[(np.fabs(eta0)<etamax) & (np.fabs(eta1)<etamax)]
    X = X[(np.fabs(eta0)<etamax) & (np.fabs(eta1)<etamax)]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    labels = id0<6

    samples = [X[(deta0>bins[i]) & (deta0<bins[i+1])] for i in range(len(bins)-1)]
    y = [labels[(deta0>bins[i]) & (deta0<bins[i+1])] for i in range(len(bins)-1)]
    output = [ [float(len(yy[yy==True]))/len(yy)]*len(yy) for yy in y]

    return samples,output,y

def getToys(means,stds,fractions):
    samplesize = 20000
    signal = []
    bckg = []
    for f in fractions:
        signal.append(
            np.stack([
                    np.random.normal(mu,sigma,samplesize*f)
                    for mu[0],sigma[0] in zip(means,stds)
                    ])
            )#signal
        bckg.append(
            np.stack([
                    np.random.normal(mu,sigma,samplesize)
                    for mu[1],sigma[1] in zip(means,stds)
                    ])
            )#signal
    y = fractions
    return samples,output,y

