from utils import evaluateModel,SetupATLAS
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from dataprovider import getSamples,getInclSamples,getToys
from models import trainqgcomplete,trainweak
from sklearn.metrics import auc

SetupATLAS()

myfraction = 0.4
layersize = 30

NB_EPOCH_weak = 150
features = ['n','w','f0']
colors = ['r','c','m']
etamax = 2.1
nbins = 12
def run(myfraction): 
    
    samples,fractions,labels = getInclSamples(features,etamax,nbins,'data/default.root')
    print 'n bins',len(labels)
    print 'sample sizes',[len(y) for y in labels]
    print 'fraction',fractions[0][0]
    
    fractions = [[myfraction for x in xx] for xx in fractions]

    print 'newfraction',fractions[0][0]
    
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
        
#### weak supervision
    model_weak = trainweak(trainsamples,trainfractions,layersize,NB_EPOCH_weak,
                               l2reg=1e-1,sdreg=1e-1,suffix='paper')

###performance
    X_train = np.concatenate( trainsamples )
    y_train = np.concatenate( trainlabels )
    X_test = np.concatenate( testsamples )
    y_test = np.concatenate( testlabels )

    #weak
    predict_weak = model_weak.predict_proba(X_test)
        
    fpr,tpr,thres = roc_curve(y_test, predict_weak)
    area =  auc(fpr, tpr)
    if area<0.5:
        predict_weak = 1-predict_weak
        fpr,tpr,thres = roc_curve(y_test, predict_weak)


#    quarks = predict_weak[y_test==True]
#    gluons = predict_weak[y_test==False]
#    plt.hist(quarks,histtype='stepfilled',normed=True,alpha=0.5,label='quarks')
#    plt.hist(gluons,histtype='stepfilled',normed=True,alpha=0.5,label='gluons')
#    plt.legend(loc='upper right',frameon=False)
#    plt.savefig('paper_plots/predictions_noflip.png')

    plt.plot(fpr, tpr, linestyle='-', label='fraction=%1.2f, AUC=%1.2f'%(myfraction,area))


    if myfraction==0.1:
        for X,f,c in zip(X_test.T,features,colors):
            fpr,tpr,thres = roc_curve(y_test, -1*X.T)
            area = auc(fpr, tpr)
            if area<0.5:
                fpr,tpr,thres = roc_curve(y_test, X.T)	
                area = auc(fpr, tpr)
            plt.plot(fpr, tpr, linestyle='--', color=c, label=f+', AUC=%1.2f'%area)
                

    #plt.clf()

myfracs = [0.1,0.23,0.4,0.5,0.7,0.9]

for myfrac in myfracs:
    run(myfrac)
plt.plot([],[],linestyle='-.',color='gray',alpha=0.5,label='training sample')
plt.legend(loc='lower right',frameon=False)
plt.savefig('paper_plots/scanfraction_newloss_newinit.png')
