import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from dataprovider import getInclToys
from models import getweak

l2regs = [0,4e-1,5e-1,6e-1]
sdregs = [0,1e-4,1e-3,1e-2]

nruns = 100
nbins = 20
layersize = 10
toymeans = [(18,26),(0.06,0.09),(0.23,0.28)]
toystds  = [(7,8),  (0.04,0.04),(0.05,0.04)]
fraction = 0.6

def run(l2reg,sdreg,run):
    samples,fractions,labels = getInclToys(toymeans,toystds,fraction,nbins)

    testsamples = []
    testlabels = []
    for X,y,f in zip(samples,labels,fractions):
        X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(X, y, f, 
                                                                             test_size=0.3)
        testsamples.append(X_test)
        testlabels.append(y_test)
    
    inputsize = testsamples[0].shape[1]
    model_weak = getweak(inputsize,layersize)
    weightname = 'output/model_run%d_nsamples20_l2reg%s_sdreg%s.h5'%(run,l2reg,sdreg)
    model_weak.load_weights(weightname)

    X_test = np.concatenate( testsamples )
    y_test = np.concatenate( testlabels )

    predict_weak = model_weak.predict_proba(X_test)
    fpr,tpr,thres = roc_curve(y_test, predict_weak)
    area =  auc(fpr, tpr)
    if area<0.5:
        fpr,tpr,thres = roc_curve(y_test, 1 - predict_weak)	
        area = auc(fpr, tpr)
    return area

import os.path

fig, axarr = plt.subplots(len(l2regs),sharex=True)

for l2reg,ax1 in zip(l2regs,axarr):
    ax2 = ax1.twinx()

    means = []
    stds = []
    for sdreg in sdregs:
        print l2reg,sdreg
        aucs = []
        for i in range(nruns):
            filename = 'output/model_run%d_nsamples20_l2reg%s_sdreg%s.h5'%(i,l2reg,sdreg)
            if os.path.exists(filename):
                aucs.append( run(l2reg,sdreg,i) )
        means.append(np.median(aucs))
        q75, q25 = np.percentile(aucs, [75 ,25])
        iqr = q75-q25
        print iqr
        stds.append(iqr)

    ax1.plot(sdregs,stds,'b',linestyle='--',marker='o')
    ax1.set_xscale('symlog',linthreshx=5e-5)
    ax1.set_ylabel('$IQR(AUC)$',color='b')
    ax1.yaxis.set_ticks(np.arange(0.,0.4,0.1))
    ax1.set_ylim([0.,0.35])
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
        
    ax2.plot(sdregs,means,'r',linestyle='--',marker='v')
    ax2.set_xscale('symlog',linthreshx=5e-5)
    ax2.set_ylabel('$<AUC>$',color='r')
    ax2.yaxis.set_ticks(np.arange(0.5,1.0,0.1))
    ax2.set_ylim([0.45,0.95])
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    
    ax1.text(0.75,0.45,'$\lambda_{L2}$: %s'%l2reg, transform=ax1.transAxes, fontsize=14)

axarr[-1].set_xlabel('$\lambda_{SD}$')

plt.savefig('paper_plots/sigmameanvsreg.png')
