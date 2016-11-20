import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from dataprovider import getInclToys
from models import trainweak

nruns = 30
layersize = 10
NB_EPOCH_weak = 30
nbins = [1,10,50,100,200]
toymeans = [(18,26),(0.06,0.09),(0.23,0.28)]
toystds  = [(7,8),  (0.04,0.04),(0.05,0.04)]
fraction = 0.6

def run(nsamples):
    samples,fractions,labels = getInclToys(toymeans,toystds,fraction,nsamples)

    trainsamples = []
    trainlabels = []
    trainfractions = []
    testsamples = []
    testlabels = []
    for X,y,f in zip(samples,labels,fractions):
        X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(X, y, f, 
                                                                             test_size=0.3)
        trainsamples.append(X_train)
        trainlabels.append(y_train)
        trainfractions.append(f_train)
        testsamples.append(X_test)
        testlabels.append(y_test)

    model_weak = trainweak(trainsamples,trainfractions,layersize,NB_EPOCH_weak,
                           l2reg=5e-1,sdreg=1e-3)

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
    aucs = [run(nsamples) for i in range(nruns)]
#    print aucs
    means.append(np.mean(aucs))
    stds.append(np.std(aucs))

plt.plot(nbins,stds)
plt.xlabel('# of samples')
plt.ylabel('$\sigma(AUC)$')
plt.savefig('paper_plots/sigmavsnsamples.png')
plt.clf()
plt.plot(nbins,means)
plt.xlabel('# of samples')
plt.ylabel('$<AUC>$')
plt.savefig('paper_plots/meanvsnsamples.png')
