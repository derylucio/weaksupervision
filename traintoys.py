import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from dataprovider import getInclToys
from models import trainweak

from optparse import OptionParser
p = OptionParser()
p.add_option('--nSamples', type = "string", default = '10', dest = 'nSamples', help = 'number of samples ')
p.add_option('--l2Reg', type = "string", default = '5e-1', dest = 'l2Reg', help = 'L2 regularization ')
p.add_option('--sdReg', type = "string", default = '1e-3', dest = 'sdReg', help = 'SD regularization ')
p.add_option('--runNumber', type = "string", default = '0', dest = 'runNumber', help = 'run number ')

(o,a) = p.parse_args()
print o

nbins = int(o.nSamples)
l2reg = float(o.l2Reg)
sdreg = float(o.sdReg)
run = int(o.runNumber)

save_prefix = 'model_run'+o.runNumber+'_nsamples'+o.nSamples+'_l2reg'+o.l2Reg+'_sdreg'+o.sdReg

layersize = 10
NB_EPOCH_weak = 30
toymeans = [(18,26),(0.06,0.09),(0.23,0.28)]
toystds  = [(7,8),  (0.04,0.04),(0.05,0.04)]
fraction = 0.6

def run(nsamples):
    samples,fractions,labels = getInclToys(toymeans,toystds,fraction,nsamples)

    trainsamples = []
    trainlabels = []
    trainfractions = []
    for X,y,f in zip(samples,labels,fractions):
        X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(X, y, f, 
                                                                             test_size=0.3)
        trainsamples.append(X_train)
        trainlabels.append(y_train)
        trainfractions.append(f_train)

    model_weak = trainweak(trainsamples,trainfractions,layersize,NB_EPOCH_weak,
                           l2reg=l2reg,sdreg=sdreg,suffix=save_prefix)

    model_weak.save_weights(save_prefix + '.h5') 

run(nbins)
