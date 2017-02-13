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

from pylab import *
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

def getAUC(predict_weak, y_test):
    fpr,tpr,thres = roc_curve(y_test, predict_weak)
    area =  auc(fpr, tpr)
    if area < 0.5:
        fpr,tpr,thres = roc_curve(y_test, 1 - predict_weak) 
        area = auc(fpr, tpr)
    return area

def evaluateModel(plot_ax, predictions, y_test, label):
    print predictions.shape
    print y_test.shape
    fpr,tpr,thres = roc_curve(y_test, predictions)	
    area =  auc(fpr, tpr)
    if area < 0.5:
        fpr,tpr,thres = roc_curve(y_test, 1 - predictions)	
        area = auc(fpr, tpr)

    if(label == 'Fully supervised NN' or label == 'Weakly supervised NN'):
        plot_ax.plot(fpr, tpr, label=label+', AUC=%1.2f'%area)
    else:
        plot_ax.plot(fpr, tpr, linestyle='--', label=label+', auc=%1.2f'%area)
    return area
