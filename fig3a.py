from utils import evaluateModel,SetupATLAS
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from dataprovider import getSamples
from models import traincomplete,trainweak
from sklearn.metrics import auc

layersize = 30

NB_EPOCH = 150
NB_EPOCH_weak = 30
LEARNING_RATE = 9e-3
features = ['n','w','f0']
etamax = 2.1
nbins = 12
bins = np.linspace(-2.1,2.1,nbins+1)

def run(): 
    
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
    model_weak = trainweak(trainsamples,trainfractions,layersize, NB_EPOCH_weak, LEARNING_RATE)

###performance
    X_test = np.concatenate( testsamples )
    y_test = np.concatenate( testlabels )

    SetupATLAS()

    predict_complete = model_complete.predict_proba(X_test)
    fpr,tpr,thres = roc_curve(y_test, predict_complete)
    area =  auc(fpr, tpr)
    plt.plot(tpr, fpr, linestyle='-', label='Fully supervised NN, AUC=%1.2f'%area)

    predict_weak = model_weak.predict_proba(X_test)
    fpr,tpr,thres = roc_curve(y_test, predict_weak)
    area =  auc(fpr, tpr)
    if area<0.5:
        fpr,tpr,thres = roc_curve(y_test, 1 - predict_weak)	
        area = auc(fpr, tpr)
    plt.plot(tpr, fpr, linestyle='-', label='Weakly supervised NN, AUC=%1.2f'%area)
    
    for X,f in zip(X_test.T,features):
        fpr,tpr,thres = roc_curve(y_test, -1*X.T)
        area = auc(fpr, tpr)
        if area<0.5:
            fpr,tpr,thres = roc_curve(y_test, X.T)	
            area = auc(fpr, tpr)
        plt.plot(tpr, fpr, linestyle='--', label=f+', AUC=%1.2f'%area)
    
    plt.ylabel('Gluon Jet efficiency')
    plt.xlabel('Quark Jet efficiency')
    plt.legend(loc='upper left',frameon=False)

    plt.ylim([0.,0.6])
    plt.xlim([0.3,0.9])

    plt.savefig('paper_plots/fig3a.png')
    plt.clf()


run()
