from rootpy.io import root_open
from root_numpy import hist2array,array2hist
import ROOT as r
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from models import trainqgcomplete,trainweak
from sklearn.metrics import auc
from utils import getjetvar
from sklearn.preprocessing import StandardScaler

filename = 'data/default160GeV.root'
histname = 'akt4TopoEM_trkwidth_vs_ntrk_PT160_eta0_q'
fraction = 0.23
nsamples = 12
features = ['n','w']
colors = ['r','c']

layersize = 30
layersize_complete = 10
NB_EPOCH = 150
NB_EPOCH_weak = 30

qmc = []
qdata = []
gmc = []
gdata = []

with root_open('Max_histos/data.root') as fdata, root_open('Max_histos/mc.root') as fmc:
    hdata = fdata.Get(histname)
    hmc = fmc.Get(histname)

    id0 = getjetvar('j0','id',filename)
    eta0 = getjetvar('j0','eta',filename)
    n0 = getjetvar('j0','ntrk',filename)
    w0 = getjetvar('j0','wtrk',filename)
    
    n0q = n0[id0<5]
    w0q = w0[id0<5]

    def getg(h):

        htmp = h.Clone("htmp")
        htmp.Reset()
        [htmp.Fill(w,n) for w,n in zip(w0q,n0q)]
        htmp.Scale(1./htmp.Integral(0,-1,0,-1))

        htmp2 = htmp.Clone("copy")
        htmp.Divide(h)
        htmp.Add(htmp2,-1)

        nx = htmp.GetNbinsX()
        ny = htmp.GetNbinsY()
        [htmp.SetBinContent(ix,iy,0.) for ix in np.arange(1,nx+1) for iy in np.arange(1,ny+1) if htmp.GetBinContent(ix,iy)<0.]
        xbin = htmp.GetXaxis().FindBin(0.15)
        ybin = htmp.GetYaxis().FindBin(20)
        [htmp.SetBinContent(ix,iy,0.) for ix in np.arange(xbin,nx+1) for iy in np.arange(1,ny+1)]
        [htmp.SetBinContent(ix,iy,0.) for ix in np.arange(1,nx+1) for iy in np.arange(ybin,ny+1)]

        return htmp
        
    def getgluons(hist,samplesize):
        w,n = r.Double(0),r.Double(0)
        g = []
        for x in range(samplesize):
            hist.GetRandom2(w,n)
            g.append( (int(n),float(w)) )            

        return np.array(g)

    def getSamples(quarks,gluons,nsamples):
        X = np.concatenate( (quarks,gluons) )
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = np.array( [True for x in quarks]+[False for x in gluons] )
        
        rndmindices = np.random.permutation(X.shape[0])
        samples = np.array_split(X[rndmindices],nsamples)
        labels = np.array_split(y[rndmindices],nsamples)
        output = [ [fraction]*len(yy) for yy in labels]
        return samples,output,labels

    gmc = getg(hmc)
    gdata = getg(hdata)
    quarks = np.vstack( [n0q,w0q] ).T
    samplesize = int(float(quarks.shape[0])/fraction)
    mcgluons = getgluons(gmc,samplesize)

    datagluons = getgluons(gdata,samplesize)

    samples,fractions,labels = getSamples(quarks,datagluons,nsamples)
    print 'n bins',len(labels)
    print 'sample sizes',[len(y) for y in labels]
    print 'fraction',fractions[0][0]

    samplesCR1,fractionsCR1,labelsCR1 = getSamples(quarks,mcgluons,nsamples)
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
    model_complete = trainqgcomplete(trainsamples,trainlabels,
                                     NB_EPOCH,layersize_complete)
    
#### weak supervision
    model_weak = trainweak(trainsamplesCR1,trainfractions,layersize,NB_EPOCH_weak,
                           l2reg=0,sdreg=1e-3,suffix='paper')

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

#    for X,f,c in zip(X_train.T,features,colors):
#        fpr,tpr,thres = roc_curve(y_train, -1*X.T)
#        area = auc(fpr, tpr)
#        if area<0.5:
#            fpr,tpr,thres = roc_curve(y_train, X.T)	
#            area = auc(fpr, tpr)
#        plt.plot(fpr, tpr, linestyle='-.', color=c, alpha=0.5)
#    
#    plt.plot([],[],linestyle='-.',color='gray',alpha=0.5,label='training sample')

    plt.xlabel('Gluon Jet efficiency')
    plt.ylabel('Quark Jet efficiency')
    plt.legend(loc='lower right',frameon=False)
    
    plt.savefig('paper_plots/test_noswap_opti.png')
    plt.clf()


#    gmc.SaveAs("test.root")

#    g = q/L - q
