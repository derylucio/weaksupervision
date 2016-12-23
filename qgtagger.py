from utils import evaluateModel #,SetupATLAS
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from dataprovider import getSamples,getToys
from models import traincomplete,trainweak

nruns = 4
layersize = 30

NB_EPOCH = 40
NB_EPOCH_weak = 50
features = ['n','w','eec2']
etamax = 2.1
nbins = 12
bins = np.linspace(-2.1,2.1,nbins+1)
usetoys = True
toymeans = [(18,26),(0.06,0.09),(0.23,0.28)]
toystds  = [(7,8),  (0.04,0.04),(0.05,0.04)]
toyfractions = [0.2, 0.9]*15 #[0.24, 0.25, 0.25, 0.26, 0.27, 0.29, 0.31, 0.33, 0.37, 0.39, 0.44]

def run(learning_rate, run=0): 
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
    model_weak = trainweak(trainsamples,trainfractions,layersize,NB_EPOCH_weak,suffix, learning_rate)
    X_test = np.concatenate( testsamples )
    y_test = np.concatenate( testlabels )
    auc_sup = None
    if run == 0:
        auc_sup = evaluateModel(None, plt, model_complete, 'CompleteSupervision', X_test, y_test)
    auc_weak = evaluateModel(None, plt, model_weak, 'WeakSupervision', X_test, y_test)
    return auc_sup, auc_weak

# SetupATLAS()
learning_rates = [9e-3]
colors = ['r', 'g', 'b', 'c', 'k', 'm']
all_weak_aucs = []
runs = range(nruns)
for index, lr in enumerate(learning_rates):
    plt.xlabel("False Positive")
    plt.ylabel("True Positive")
    plt.ylim([0,1.7])
    aucs_weak = []
    aucs_sup = []
    for runnum in runs:
        auc_sup,auc_weak = run(lr, run=runnum)
        aucs_sup.append(auc_sup)
        aucs_weak.append(auc_weak)
    all_weak_aucs.append(aucs_weak)
    plt.legend(loc='upper left', title='Trained on Data Fractions [0.2, 0.7]', frameon=True)
    plt.savefig('plots/WeakSupervision' + str(index))
    plt.close()
plt.xlabel("run")
plt.ylabel("auc")
for index, lr in enumerate(learning_rates):
    plt.plot(all_weak_aucs[index], color=colors[index], label=str(lr))
plt.legend(loc='lower right', frameon=True)
plt.savefig('plots/stability_lr_study')
# plt.plot(runs,aucs_sup,label='complete supervision')
# plt.plot(runs,aucs_weak,label='weak supervision')
# plt.ylabel('AUC')
# plt.xlabel('run')
# plt.legend(loc='upper left',title='hidden layer size: %d'%layersize)
# plt.savefig('plots/summary_auc_%d'%layersize)

