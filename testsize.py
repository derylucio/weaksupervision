import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from dataprovider import getInclToys
from models import getweak

nruns = 100
nbins = [1,3,5,8,10,12,15,18,20,25,30,50,100]
layersize = 10
toymeans = [(18,26),(0.06,0.09),(0.23,0.28)]
toystds  = [(7,8),  (0.04,0.04),(0.05,0.04)]
fraction = 0.6

def run(nsamples,run):
    samples,fractions,labels = getInclToys(toymeans,toystds,fraction,nsamples)

    testsamples = []
    testlabels = []
    for X,y,f in zip(samples,labels,fractions):
        X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(X, y, f, 
                                                                             test_size=0.3)
        testsamples.append(X_test)
        testlabels.append(y_test)
    
    inputsize = testsamples[0].shape[1]
    model_weak = getweak(inputsize,layersize)
    weightname = 'output_size/model_run%d_nsamples%d_l2reg5e-1_sdreg1e-3.h5'%(run,nsamples)
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

means = []
stds = []
for nsamples in nbins:
    print nsamples
    aucs = [run(nsamples,i) for i in range(nruns)]
#    print aucs
    means.append(np.median(aucs))
    q75, q25 = np.percentile(aucs, [75 ,25])
    iqr = q75-q25
    print iqr
    stds.append(iqr)

fig, ax1 = plt.subplots()

ax1.plot(nbins,stds,'b--',marker='o')
ax1.set_xlabel('# of samples')
ax1.set_xlim([0, 110])
ax1.set_ylabel('$IQR(AUC)$',color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')

ax2 = ax1.twinx()
ax2.plot(nbins,means,'r--',marker='v')
ax2.set_ylabel('$<AUC>$',color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')

plt.savefig('paper_plots/sigmameanvsnsamples.png')
