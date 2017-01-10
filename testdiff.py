import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from models import trainweak
from dataprovider import getToys
from sklearn.cross_validation import train_test_split
from sklearn.mixture import GaussianMixture

nruns = 20
diffs = [0.6, 0.2]#np.linspace(0.1, 0.75, 14)
layersize = 30
toymeans = [(18,26),(0.06,0.09),(0.23,0.28)]
toystds  = [(7,8),  (0.04,0.04),(0.05,0.04)]
start_fraction = 0.2
num_samples = 15
etamax = 2.1
nbins = 12
bins = np.linspace(-2.1,2.1,nbins+1)
NB_EPOCH_weak = 25
LEARNING_RATE = 9e-3
GMM = GaussianMixture(2)
area_before = []
area_after = []

def getAUC(predict_weak, y_test):
    fpr,tpr,thres = roc_curve(y_test, predict_weak)
    area =  auc(fpr, tpr)
    if area < 0.5:
        fpr,tpr,thres = roc_curve(y_test, 1 - predict_weak) 
        area = auc(fpr, tpr)
    print area
    return area

def run(diff,run):
    toyfractions = [start_fraction, start_fraction + diff]*num_samples
    print toyfractions
    suffix = '_etamax%d_nbins%d_diff%d_reg'%(etamax*10,nbins,diff)
    samples,fractions,labels = getToys(toymeans,toystds,toyfractions)
    trainsamples = []
    trainlabels = []
    trainfractions = []
    testsamples = []
    testlabels = []
    for X,y,f in zip(samples,labels,fractions):
        trainsamples.append(X)
        trainlabels.append(y)
        trainfractions.append(f)
    X_test, _, y_test = getToys(toymeans,toystds,[0.5, 0.5])
    testsamples.append(np.concatenate(X_test))
    testlabels.append(np.concatenate(y_test))

#### weak supervision
    model_weak = trainweak(trainsamples,trainfractions,layersize, NB_EPOCH_weak, suffix, LEARNING_RATE)
    X_test = np.concatenate( testsamples )
    y_test = np.concatenate( testlabels )
    predict_weak = model_weak.predict_proba(X_test)
    area = getAUC(predict_weak, y_test)
    area_before.append(area)
    GMM.fit(predict_weak)
    predict_weak = GMM.predict_proba(predict_weak)
    print predict_weak[:5]
    predict_weak = predict_weak[:, 0]
    area = getAUC(predict_weak, y_test)
    area_after.append(area)
    # plt.hist(predict_weak[y_test == 1], histtype="step", normed=True, label='Quark')
    # plt.hist(predict_weak[y_test == 0], histtype="step", normed=True, label='Gloun')
    # plt.legend(loc='lower right', frameon=True)
    # plt.savefig('plots/dist' + str(run))
    # plt.close()
    return area

medians = []
iqrs = []
for diff in diffs:
    print diff
    aucs = [run(diff,i) for i in range(nruns)]
#    print aucs
    print "<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>"
    print diff, aucs
    print "<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>"
    medians.append(np.median(aucs))
    q75, q25 = np.percentile(aucs, [75 ,25])
    iqr = q75-q25
    print iqr
    iqrs.append(iqr)

fig, ax1 = plt.subplots()

ax1.plot(diffs, iqrs,'b--',marker='o')
ax1.set_xlabel('Difference Between Fractions')
ax1.set_xlim([0, 1])
ax1.set_ylabel('$IQR(AUC)$',color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')

ax2 = ax1.twinx()
ax2.plot(diffs, medians,'r--',marker='v')
ax2.set_ylabel('$<AUC>$',color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')

plt.savefig('plots/sigmameanvsfractiondiff_noflip_gmm.png')
plt.plot(area_before, color='r', label='Before GMM')
plt.plot(area_after, color='b', label='After GMM')
plt.legend()
plt.show()