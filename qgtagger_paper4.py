from utils import evaluateModel,SetupATLAS
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from dataprovider import getSamples,getInclSamples,getToys
from models import trainqgcomplete,trainweak
from sklearn.metrics import auc

SetupATLAS()

#corr = np.array([1,0.01,0.])
corr = np.array([1,0.01])
gcorr = np.array([0.5,0.01])

layersize = 30

NB_EPOCH = 300
NB_EPOCH_weak = 30
#features = ['n','w','f0']
features = ['n','w']
colors = ['r','c','m']
etamax = 2.1
nbins = 12
def run(): 
    
    samples,fractions,labels = getInclSamples(features,etamax,nbins,'data/default.root')
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
        X_train, X_test, y_train, y_test = train_test_split(X+corr*y[:,np.newaxis]+corr*~y[:,np.newaxis]*np.random.choice([0,1],size=y.shape)[:,np.newaxis]
                                                            , y, test_size=0.3)
        trainsamples.append(X_train)
        trainlabels.append(y_train)

    trainlabelsCR1 = []
    trainsamplesCR1 = []
    trainfractions = []
    testsamplesCR1 = []
    testlabelsCR1 = []
    for X,y,f in zip(samplesCR1,labelsCR1,reshaped_f):
        X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(X#-corr*y[:,np.newaxis]+gcorr*~y[:,np.newaxis]
                                                                             , y, f, test_size=0.3)
        trainlabelsCR1.append(y_train)
        trainsamplesCR1.append(X_train)
        trainfractions.append(f_train)
        testsamplesCR1.append(X_test)
        testlabelsCR1.append(y_test)

    model_completes = []
    model_completeCR1s = []
    model_weaks = []
    for x in range(30):
### complete supervision
        model_complete = trainqgcomplete(trainsamples,trainlabels,NB_EPOCH)
        model_completeCR1 = trainqgcomplete(trainsamplesCR1,trainlabelsCR1,NB_EPOCH)
        
#### weak supervision
        model_weak = trainweak(trainsamplesCR1,trainfractions,layersize,NB_EPOCH_weak,
                               l2reg=1e-1,sdreg=1e-1,suffix='paper')
        model_completes.append( model_complete )
        model_completeCR1s.append( model_completeCR1 )
        model_weaks.append( model_weak )

###performance
    X_train = np.concatenate( trainsamples )
    y_train = np.concatenate( trainlabels )
    X_test = np.concatenate( testsamplesCR1 )
    y_test = np.concatenate( testlabelsCR1 )

    effs = [0.3,0.5,0.7,0.9]

    def geteffcut(jets,wp=0.5):
        
        varmin = jets.mean()-2*jets.std()
        varmax = jets.mean()+2*jets.std()
        print varmax,varmin
        wpcut = 0.
        cuts = np.arange(varmin,varmax,float(varmax)/5000)
        for cut in cuts:
            eff = float(jets[jets>cut].shape[0])/jets.shape[0]
            if eff<wp:
                #preveff = float(jets[jets>(cut)].shape[0])/jets.shape[0]
                wpcut = cut #if np.fabs(wp-eff)>np.fabs(wp-preveff) else (cut-1)
                break
        print 'eff,wpcut',eff,wpcut
        return wpcut

    allgeffs = []
    allgeffsCR1 = []
    allgeffs_weak = []
    for model_complete, model_completeCR1, model_weak in zip(model_completes,model_completeCR1s,model_weaks):
    #complete
        predict_complete = model_complete.predict_proba(X_test)

        quarks = predict_complete[y_test==True]
        gluons = predict_complete[y_test==False]
        
        geffs = []
        for eff in effs:
            cut = geteffcut(quarks,eff)
            geffs.append( float(gluons[gluons>cut].shape[0])/gluons.shape[0] )
        geffs = np.array(geffs)

    #completeCR1
        predict_complete = model_completeCR1.predict_proba(X_test)

        quarks = predict_complete[y_test==True]
        gluons = predict_complete[y_test==False]

        geffsCR1 = []
        for eff in effs:
            cut = geteffcut(quarks,eff)
            geffsCR1.append( float(gluons[gluons>cut].shape[0])/gluons.shape[0] )
        geffsCR1 = np.array(geffsCR1)
    
    #weak
        predict_weak = model_weak.predict_proba(X_test)
        
        fpr,tpr,thres = roc_curve(y_test, predict_weak)
        area =  auc(fpr, tpr)
        if area<0.5:
            predict_weak = 1-predict_weak

        quarks = predict_weak[y_test==True]
        gluons = predict_weak[y_test==False]

        geffs_weak = []
        for eff in effs:
            cut = geteffcut(quarks,eff)
            geffs_weak.append( float(gluons[gluons>cut].shape[0])/gluons.shape[0] )
        geffs_weak = np.array(geffs_weak)
    
        allgeffs.append( geffs )
        allgeffsCR1.append( geffsCR1 )
        allgeffs_weak.append( geffs_weak )

    print allgeffs
    print allgeffsCR1
    print allgeffs_weak

    allgeffs      = np.array(allgeffs     )     
    allgeffsCR1   = np.array(allgeffsCR1  )
    allgeffs_weak = np.array(allgeffs_weak)

    fig, (ax1,ax2) = plt.subplots(2,sharex=True,gridspec_kw = {'height_ratios':[2, 1]})

    geffs = np.median(allgeffs,axis=0)    
    geffsCR1 = np.median(allgeffsCR1,axis=0)
    geffs_weak = np.median(allgeffs_weak,axis=0)

    def getIQR(effs):
        q75, q25 = np.percentile(effs, [75 ,25])
        return q75-q25

    geffs_err = getIQR(allgeffs)
    geffsCR1_err = getIQR(allgeffsCR1)
    geffs_weak_err = getIQR(allgeffs_weak)

    geffs_ratio = np.median(allgeffs/allgeffsCR1)
    geffs_ratio_weak = np.median(allgeffs_weak/allgeffsCR1)
    geffs_ratio_err = getIQR(allgeffs/allgeffsCR1)
    geffs_ratio_weak_err = getIQR(allgeffs_weak/allgeffsCR1)

    ax1.plot(effs, geffs     , linestyle='-', color='b', marker='o', mec='b', mfc='b', label='Fully supervised DT')
    ax1.plot(effs, geffsCR1  , linestyle='--', color='b', marker='o', mec='b', mfc='None', label='Fully supervised ST')
    ax1.plot(effs, geffs_weak, linestyle='-', color='r', marker='o', mec='r', mfc='r', label='Weakly supervised')
    ax1.yaxis.set_ticks(np.arange(0.,1.0,0.2))
    ax1.set_ylim([0.,0.8])
    ax1.set_ylabel('Gluon Jet efficiency')
    ax1.legend(loc='upper left',frameon=False)

    ax2.errorbar(effs,geffs/geffsCR1     , yerr=geffs_ratio_err      , linestyle='-', color='b', marker='o', mfc='b')
    ax2.errorbar(effs,geffs_weak/geffsCR1, yerr=geffs_ratio_weak_err , linestyle='-', color='r', marker='o', mfc='r')
    ax2.yaxis.set_ticks(np.arange(0.9,1.4,0.1))
    ax2.set_ylim([0.9,1.35])
    ax2.axhline(y=1,linestyle='--',color='black')
    ax2.set_ylabel('Ratio to FS ST')
    
    ax2.set_xlabel('Quark Jet efficiency')
    plt.xlim([0.2,1.0])

    plt.savefig('paper_plots/qgrocs4.png')
    #plt.clf()

run()
