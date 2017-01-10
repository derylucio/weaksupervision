from utils import evaluateModel #,SetupATLAS
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from dataprovider import getSamples,getToys
from models import traincomplete,trainweak

nruns = 50
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
toyfractions =  [0.1 , 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]*2 #[0.24, 0.25, 0.25, 0.26, 0.27, 0.29, 0.31, 0.33, 0.37, 0.39, 0.44]

def run(learning_rate, nrun=0): 
    suffix = '_etamax%d_nbins%d_run%d'%(etamax*10,nbins, nrun)
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
        # X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(X, y, f, test_size=0.3)
        trainsamples.append(X)
        trainlabels.append(y)
        trainfractions.append(f)
    
    test_samples1, _, test_labels1 = getToys(toymeans,toystds,[0.5, 0.5])
    test_samples2, _, test_labels2 = getToys(toymeans,toystds,[0.2, 0.8])
    
#### weak supervision
    model_weak = trainweak(trainsamples,trainfractions,layersize,NB_EPOCH_weak,suffix, learning_rate)
    X_test1 = np.concatenate( test_samples1 )
    y_test1 = np.concatenate( test_labels1 )
    X_test2 = np.concatenate( test_samples2 )
    y_test2 = np.concatenate( test_labels2 )
    auc_sup = None
    print 'Lengths : ', len(y_test1), len(y_test2)
    print 'Fracs : ', sum(y_test1), sum(y_test2)
    if nrun == 0:
        ### complete supervision
        model_complete = traincomplete(trainsamples,trainlabels,NB_EPOCH)
        auc_sup = evaluateModel(None, plt, model_complete, 'CompleteSupervision55', X_test1, y_test1)
        auc_sup = evaluateModel(None, plt, model_complete, 'CompleteSupervision28', X_test2, y_test2)
    auc_weak = evaluateModel(None, plt, model_weak, 'WeakSupervision55', X_test1, y_test1)
    auc_weak = evaluateModel(None, plt, model_weak, 'WeakSupervision28', X_test2, y_test2)
    return auc_sup, auc_weak

# SetupATLAS()
learning_rates = [1e-3]
colors = ['r', 'g', 'b', 'c', 'k', 'm']
all_weak_aucs = []
runs = range(nruns)
for index, lr in enumerate(learning_rates):
    plt.xlabel("False Positive")
    plt.ylabel("True Positive")
    plt.ylim([0,3])
    for runnum in runs:
        print "this is run : ", runnum
        auc_sup,auc_weak = run(lr, nrun=runnum)
        all_weak_aucs.append(auc_weak)
        if runnum == 2:
            plt.legend(loc='upper left', title='Trained on Data Fractions Uniform Shifted', frameon=True)
            plt.savefig('plots/WeakSupervision_TT_check_shifted')
            plt.close()
plt.close()
plt.xlabel("run")
plt.ylabel("auc")
plt.plot(all_weak_aucs)
plt.legend(loc='lower right', frameon=True)
plt.savefig('plots/stability_uniform-Shifted')
# plt.plot(runs,aucs_sup,label='complete supervision')
# plt.plot(runs,aucs_weak,label='weak supervision')
# plt.ylabel('AUC')
# plt.xlabel('run')
# plt.legend(loc='upper left',title='hidden layer size: %d'%layersize)
# plt.savefig('plots/summary_auc_%d'%layersize)

