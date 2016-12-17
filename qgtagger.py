from utils import evaluateModel #,SetupATLAS
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from dataprovider import getSamples,getToys
from models import traincomplete,trainweak

nruns = 5
layersize = 30

NB_EPOCH = 40
NB_EPOCH_weak = 30
features = ['n','w','eec2']
etamax = 2.1
nbins = 12
bins = np.linspace(-2.1,2.1,nbins+1)
usetoys = True
toymeans = [(18,26),(0.06,0.09),(0.23,0.28)]
toystds  = [(7,8),  (0.04,0.04),(0.05,0.04)]
toyfractions = [0.24, 0.25, 0.25, 0.26, 0.27, 0.29, 0.31, 0.33, 0.37, 0.39, 0.44]

def run(run=0): 
    
    suffix = '_etamax%d_nbins%d_run%d'%(etamax*10,nbins,run)
    if usetoys:
        samples,fractions,labels = getToys(toymeans,toystds,toyfractions)
    else:
        samples,fractions,labels = getSamples(features,etamax,bins)
    print 'n bins',len(labels)
    print 'sample sizes',[len(y) for y in labels]

    trainsamples = []
    trainlabels = []
    trainfractions = []
    testsamples = []
    testlabels = []
    for X,y,f in zip(samples,labels,fractions):
        X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(X, y, f, test_size=0.3)
        trainsamples.append(X_train)
        trainlabels.append(y_train)
        trainfractions.append(f_train)
        testsamples.append(X_test)
        testlabels.append(y_test)

### complete supervision
    model_complete = traincomplete(trainsamples,trainlabels,NB_EPOCH)
    
#### weak supervision
    model_weak = trainweak(trainsamples,trainfractions,layersize,NB_EPOCH_weak,suffix)
    X_test = np.concatenate( testsamples )
    y_test = np.concatenate( testlabels )
    auc_sup = None
    if run == 0:
        auc_sup = evaluateModel(None, plt, False, model_complete, 'Complete Supervision', X_test, y_test)
    auc_weak = evaluateModel(None, plt, False, model_weak, 'Weak Supervision', X_test, y_test)
    return auc_sup, auc_weak

# SetupATLAS()
aucs_sup = []
aucs_weak = []
runs = range(nruns)
plt.xlabel("False Positive")
plt.ylabel("True Positive")
plt.ylim([0,1.7])
for runnum in runs:
    auc_sup,auc_weak = run(runnum)
    aucs_sup.append(auc_sup)
    aucs_weak.append(auc_weak)

plt.legend(loc='upper left', frameon=True)
plt.savefig('plots/No-Regularization-Flipped')

# plt.plot(runs,aucs_sup,label='complete supervision')
# plt.plot(runs,aucs_weak,label='weak supervision')
# plt.ylabel('AUC')
# plt.xlabel('run')
# plt.legend(loc='upper left',title='hidden layer size: %d'%layersize)
# plt.savefig('plots/summary_auc_%d'%layersize)

