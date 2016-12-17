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
features = ['n','w','f0']
colors = ['r','c','m']
etamax = 2.1
nbins = 12
def run(): 
    
    samples,fractions,labels = getInclSamples(features,etamax,nbins,'data/default5GeV.root')
    print 'n bins',len(labels)
    print 'sample sizes',[len(y) for y in labels]
    print 'fraction',fractions[0][0]

    samplesCR1,fractionsCR1,labelsCR1 = getInclSamples(features,etamax,nbins,'data/default.root')
    print 'n bins',len(labelsCR1)
    print 'sample sizes',[len(y) for y in labelsCR1]
    print 'fraction',fractionsCR1[0][0]

    reshaped_f = [ [fractions[0][0]]*len(yy) for yy in labelsCR1]

    trainsamples = []
    trainlabels = []
    for X,y in zip(samples,labels):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        trainsamples.append(X_train)
        trainlabels.append(y_train)

    trainsamplesCR1 = []
    trainfractions = []
    testsamplesCR1 = []
    testlabelsCR1 = []
    for X,y,f in zip(samplesCR1,labelsCR1,reshaped_f):
        X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(X, y, f, test_size=0.3)
        trainsamplesCR1.append(X_train)
        trainfractions.append(f_train)
        testsamplesCR1.append(X_test)
        testlabelsCR1.append(y_test)

### complete supervision
    model_complete = trainqgcomplete(trainsamples,trainlabels,NB_EPOCH)
    
#### weak supervision
    model_weak = trainweak(trainsamplesCR1,trainfractions,layersize,NB_EPOCH_weak,
                           l2reg=5e-1,sdreg=1e-3,suffix='paper')

###performance
    X_train = np.concatenate( trainsamples )
    y_train = np.concatenate( trainlabels )
    X_test = np.concatenate( testsamplesCR1 )
    y_test = np.concatenate( testlabelsCR1 )

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
    
    for X,f,c in zip(X_test.T,features,colors):
        fpr,tpr,thres = roc_curve(y_test, -1*X.T)
        area = auc(fpr, tpr)
        if area<0.5:
            fpr,tpr,thres = roc_curve(y_test, X.T)	
            area = auc(fpr, tpr)
        plt.plot(fpr, tpr, linestyle='--', color=c, label=f+', AUC=%1.2f'%area)

    for X,f,c in zip(X_train.T,features,colors):
        fpr,tpr,thres = roc_curve(y_train, -1*X.T)
        area = auc(fpr, tpr)
        if area<0.5:
            fpr,tpr,thres = roc_curve(y_train, X.T)	
            area = auc(fpr, tpr)
        plt.plot(fpr, tpr, linestyle='-.', color=c, alpha=0.5)
    
    plt.plot([],[],linestyle='-.',color='gray',alpha=0.5,label='training sample')

    plt.xlabel('Gluon Jet efficiency')
    plt.ylabel('Quark Jet efficiency')
    plt.legend(loc='lower right',frameon=False)

    plt.savefig('paper_plots/qgrocs2.png')
    plt.clf()


run()
