from sklearn.preprocessing import StandardScaler
from utils import getjetvar
import numpy as np

def getSamples(features,etamax,bins):
    filename = 'data/20161103_16h20min.root'
    id0 = getjetvar('j0','id',filename)
    eta0 = getjetvar('j0','eta',filename)
    eta1 = getjetvar('j1','eta',filename)
    deta0 = getjetvar('j0','deta',filename)

    xx = [getjetvar('j0',var,filename) for var in features]
    X = np.vstack( [xx] ).T

    id0 = id0[(np.fabs(eta0)<etamax) & (np.fabs(eta1)<etamax)]
    deta0 = deta0[(np.fabs(eta0)<etamax) & (np.fabs(eta1)<etamax)]
    X = X[(np.fabs(eta0)<etamax) & (np.fabs(eta1)<etamax)]
    print X[id0<6].mean(axis=0),X[id0<6].std(axis=0)
    print X[id0==21].mean(axis=0),X[id0==21].std(axis=0)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    labels = id0<6

    samples = [X[(deta0>bins[i]) & (deta0<bins[i+1])] for i in range(len(bins)-1)]
    y = [labels[(deta0>bins[i]) & (deta0<bins[i+1])] for i in range(len(bins)-1)]
    output = [ [float(len(yy[yy==True]))/len(yy)]*len(yy) for yy in y]

    return samples,output,y

def getInclToys(means,stds,fraction,nsamples):
    samplesize = 200000
    samples = []
    labels = []
    scaler = StandardScaler()
    signal = np.stack([
            np.random.normal(mu[0],sigma[0],int(samplesize*fraction))
            for mu,sigma in zip(means,stds)
            ]).T
    bckg = np.stack([
            np.random.normal(mu[1],sigma[1],int(samplesize*(1-fraction)))
            for mu,sigma in zip(means,stds)
            ]).T
    X = np.concatenate([signal,bckg])
    X = scaler.fit_transform(X)
    y = np.array( [True for x in signal]+[False for x in bckg] )

    rndmindices = np.random.permutation(X.shape[0])
    samples = np.array_split(X[rndmindices],nsamples)
    labels = np.array_split(y[rndmindices],nsamples)
    output = [ [fraction]*len(yy) for yy in labels]

    return samples,output,labels

def getToys(means,stds,fractions):
    samplesize = 20000
    samples = []
    labels = []
    scaler = StandardScaler()
    for f in fractions:
        signal = np.stack([
                np.random.normal(mu[0],sigma[0],int(samplesize*f))
                for mu,sigma in zip(means,stds)
                ]).T
        bckg = np.stack([
                np.random.normal(mu[1],sigma[1],int(samplesize*(1-f)))
                for mu,sigma in zip(means,stds)
                ]).T
        X = np.concatenate([signal,bckg])
        samples.append( X )
        y = np.array( [True for x in signal]+[False for x in bckg] )
        labels.append(y)
    output = [ [float(len(yy[yy==True]))/len(yy)]*len(yy) for yy in labels]
    return samples,output,labels

