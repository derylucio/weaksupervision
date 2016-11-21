from utils import evaluateModel,SetupATLAS
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from dataprovider import getSamples,getInclSamples,getToys
from models import trainqgcomplete,trainweak
from sklearn.metrics import auc

layersize = 30

NB_EPOCH = 150
NB_EPOCH_weak = 30
features = ['n','w','neff']
etamax = 2.1
nbins = 12
def run(): 
    
    samples,fractions,labels = getInclSamples(features,etamax,nbins)
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
    model_complete = trainqgcomplete(trainsamples,trainlabels,NB_EPOCH)
    
#### weak supervision
    model_weak = trainweak(trainsamples,trainfractions,layersize,NB_EPOCH_weak,'paper')

###performance
    X_test = np.concatenate( testsamples )
    y_test = np.concatenate( testlabels )

    predict_complete = model_complete.predict_proba(X_test)
    fpr,tpr,thres = roc_curve(y_test, predict_complete)
    area =  auc(fpr, tpr)
    plt.plot(fpr, tpr, linestyle='-', label='Fully supervised NN, AUC=%1.2f'%area)

    predict_weak = model_weak.predict_proba(X_test)
    fpr,tpr,thres = roc_curve(y_test, predict_weak)
    area =  auc(fpr, tpr)
    if area<0.5:
        fpr,tpr,thres = roc_curve(y_test, 1 - predict_weak)	
        area = auc(fpr, tpr)
    plt.plot(fpr, tpr, linestyle='-', label='Weakly supervised NN, AUC=%1.2f'%area)
    
    for X,f in zip(X_test.T,features):
        fpr,tpr,thres = roc_curve(y_test, -1*X.T)
        area = auc(fpr, tpr)
        plt.plot(fpr, tpr, linestyle='--', label=f+', AUC=%1.2f'%area)
    
    plt.xlabel('Gluon Jet efficiency')
    plt.ylabel('Quark Jet efficiency')
    plt.legend(loc='lower right',frameon=False)

    plt.savefig('paper_plots/qgrocs.png')
    plt.clf()


run()
