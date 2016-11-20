import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from dataprovider import getInclToys
from models import trainweak

nruns = 30
layersize = 10
NB_EPOCH_weak = 30
nsamples = 20
toymeans = [(18,26),(0.06,0.09),(0.23,0.28)]
toystds  = [(7,8),  (0.04,0.04),(0.05,0.04)]
fraction = 0.6

l2regs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
sdregs = [0,1e-4,1e-3,1e-2]

def run(l2reg,sdreg):
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
                           l2reg=l2reg,sdreg=sdreg)

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
sdreg = 1e-3
for l2reg in l2regs:
    print l2reg
    aucs = [run(l2reg,sdreg) for l2reg in l2regs]
#    print aucs
    means.append(np.mean(aucs))
    stds.append(np.std(aucs))

plt.plot(l2regs,stds)
plt.xlabel('l2reg')
plt.ylabel('$\sigma(AUC)$')
plt.savefig('paper_plots/sigmavsl2reg.png')
plt.clf()

plt.plot(l2regs,means)
plt.xlabel('l2reg')
plt.ylabel('$<AUC>$')
plt.savefig('paper_plots/meanvsl2reg.png')
plt.clf()
