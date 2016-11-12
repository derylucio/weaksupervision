import rootpy
from rootpy.plotting.style import get_style, set_style
from matplotlib import rc
def SetupATLAS():
    rootpy.log.basic_config_colorized()
    #use latex for text
    rc('text', usetex=True)
    # set the style
    style = get_style('ATLAS')
    style.SetEndErrorSize(3)
    set_style(style)
    set_style('ATLAS',mpl=True)
    rc('font',family='sans-serif',weight='medium',size=16)
    rc('legend',fontsize=16)

from root_numpy import root2rec
import numpy as np
def getvar(var,filename):
    leaves = [var]
    array = root2rec(filename,'tree',leaves)
    vars = array[var]
    return vars

def getjetvar(jet,var,filename,
              ptmin=20,ptmax=200,
              etamin=0.,etamax=2.1,nocut=True):
    
    leaves = [jet+'pt',jet+'eta']
    if var not in ['pt','eta']: leaves += [jet+var]
    array = root2rec(filename,'tree',leaves)
    
    vars = array[jet+var]
    pt = array[jet+'pt']
    eta = array[jet+'eta']

    if not nocut:
        vars = vars[(pt>ptmin) & (pt<ptmax) & (np.fabs(eta)>etamin) & (np.fabs(eta)<etamax)]
    return vars

def getflavfrac(id0,id1,flavpair='gg'):
    numerator = 0.
    if 'gg'==flavpair:
        numerator = id0[ (id0==id1) & (id0==21) ].shape[0]
    elif 'gq'==flavpair or 'qg'==flavpair:
        numerator = id0[ (id0!=id1) & ((id0==21) | (id1==21)) ].shape[0]
    elif 'qq'==flavpair:
        numerator = id0[ (id0==id1) & (id0<6) ].shape[0]
    denominator = id0.shape[0]
    return float(numerator)/denominator

def getpairflav(id0,id1):
    flavs = []
    for i0,i1 in zip(id0,id1):
        if i0==i1 and 21==i0:
            flavs.append('gg')
        elif i0==i1 and i0<6:
            flavs.append('qq')
        elif i0!=i1 and (21==i0 or 21==i1):
            flavs.append('qg')
        else:
            flavs.append('idk')
    return np.array(flavs)

from pylab import *
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
def evaluateModel(hist_ax, plot_ax, model, label, x_test, y_test):
    predict_proba = model.predict_proba(x_test)
    print label, 'min', min(predict_proba)
    print label, 'max', max(predict_proba)
    print label,  'mean', np.mean(predict_proba)
    fpr,tpr,thres = roc_curve(y_test, predict_proba)	
    area =  auc(fpr, tpr)
    if area<0.5:
        fpr,tpr,thres = roc_curve(y_test, 1 - predict_proba)	
        area = auc(fpr, tpr)
    hist_ax.hist(predict_proba[y_test == 1], histtype='step', normed=True, 
                  label = 'signal')
    hist_ax.hist(predict_proba[y_test == 0], histtype='step',  normed=True, 
                  label = 'background')
    hist_ax.legend(title=label)
    print label, area
    plot_ax.plot(fpr, tpr, label=label+', auc=%1.2f'%area)
    plot_ax.legend(loc='lower right')
    return area
