import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from models import trainweak
from dataprovider import getToys
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

nruns = 50
nb = range(1, 10) #np.linspace(0.1, 0.75, 14)
layersize = 30
toymeans = [(18,26),(0.06,0.09),(0.23,0.28)]
toystds  = [(7,8),  (0.04,0.04),(0.05,0.04)]
start_fraction = 0.2
etamax = 2.1
nbins = 12
bins = np.linspace(-2.1,2.1,nbins+1)
NB_EPOCH_weak = 30 
LEARNING_RATE = 1e-3 
scaler = StandardScaler()
fractions = [[0.2, 0.22], [0.2, 0.4]]

def run(samples, fracs, X_test, y_test):
    suffix = '_batches_etamax%d_nbins%d'%(etamax*10,nbins)

    trainsamples = []
    trainfractions = []
    for X,f in zip(samples,fracs):
        trainsamples.append(X)
        trainfractions.append(f)

#### weak supervision
    model_weak = trainweak(trainsamples,trainfractions,layersize, NB_EPOCH_weak, suffix, LEARNING_RATE)

    predict_weak = model_weak.predict_proba(X_test)
    fpr,tpr,thres = roc_curve(y_test, predict_weak)
    area =  auc(fpr, tpr)
    if area < 0.5:
        fpr,tpr,thres = roc_curve(y_test, 1 - predict_weak) 
        area = auc(fpr, tpr)

    print 'Auc : ', area
    return area

all_medians = []
all_iqrs = []
for frac in fractions: 
    medians = []
    iqrs = []
    for num_batches in nb:

        print num_batches
        toyfractions = frac*num_batches
        samples,fracs,labels = getToys(toymeans,toystds,toyfractions)
        X_test, _, y_test = getToys(toymeans,toystds,[0.5, 0.5])
        X_test = np.concatenate(X_test)
        y_test = np.concatenate(y_test)
        all_samples = []
        all_samples.extend(samples)
        all_samples.append(X_test)
        all_samples = np.concatenate(all_samples)
        all_samples = scaler.fit_transform(all_samples)
        curr_prev = 0
        curr_next = 0
        for i in range(len(samples)):
            curr_next = curr_prev + len(samples[i])
            samples[i] = all_samples[curr_prev:curr_next]
            curr_prev = curr_next
        X_test = np.array(all_samples[curr_next:])

        aucs = []
        for i in range(nruns):
            print 'Run ', i
            aucs.append(run(samples, fracs, X_test, y_test))

        medians.append(np.median(aucs))
        q75, q25 = np.percentile(aucs, [75 ,25])
        iqr = q75-q25
        iqrs.append(iqr)

        print "<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>"
        print num_batches, np.median(aucs), iqr, aucs
        print "<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>"

    all_medians.append(medians)
    all_iqrs.append(iqrs)

nb = np.array(nb)*2
fig, ax1 = plt.subplots()
ax1.plot(nb, all_iqrs[0],'b--',marker='o', label='[0.2, 0.22]')
ax1.plot(nb, all_iqrs[1],'b--',marker='v')
ax1.set_xlabel('Num Batches')
ax1.set_ylabel('$IQR(AUC)$',color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')

ax2 = ax1.twinx()
ax2.plot(nb, all_medians[0],'r--',marker='o')
ax2.plot(nb, all_medians[1],'r--',marker='v', label='[0.2, 0.4]')
ax2.set_ylabel('$<AUC>$',color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
plt.savefig('toy_plots/sigmameanvsbatchsize.png')