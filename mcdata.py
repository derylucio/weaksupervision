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

filename = 'data/default.root'
histname = 'akt4TopoEM_trkwidth_vs_ntrk_PT40_eta0_q'
samplesize = 100000
f = 0.23
nsamples = 12
features = ['w','n']
colors = ['r','c','m']

layersize = 30
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
    n0 = getjetvar('j0','n',filename)
    w0 = getjetvar('j0','w',filename)
    
    n0q = n0[id0<5]
    w0q = w0[id0<5]

    def getg(h):
        a,edges = hist2array(h,return_edges=True)
        q = np.histogram2d( w0q, n0q, bins=edges, normed=True )[0]
        g = q/a - q
        return np.nan_to_num(g)
    
    gmc = getg(hmc)
    gdata = getg(hdata)
    
#    print gmc
#    print gdata

    def getnandw(g,h):
        h.Print()
        htmp = h.Clone()
        htmp = array2hist(g,htmp)
        htmp.Print()
        w,n = r.Double(0),r.Double(0)
        htmp.GetRandom2(w,n)
        print w,n

    getnandw(gmc,hmc)
#    g = q/L - q
