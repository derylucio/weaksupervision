import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from models import trainweak, traincomplete
from dataprovider import getToys
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import getAUC,SetupATLAS

nruns = 100
diffs = [0.02, 0.05, 0.1, 0.3, 0.5]
layersize = 30
toymeans = [(18,26),(0.06,0.09),(0.23,0.28)]
toystds  = [(7,8),  (0.04,0.04),(0.05,0.04)]
start_fraction = 0.2
num_samples = 1
etamax = 2.1
nbins = 12
bins = np.linspace(-2.1,2.1,nbins+1)
NB_EPOCH_weak = 25
LEARNING_RATE = 9e-3
scaler = StandardScaler()

def run(samples, fracs, X_test, y_test):
    suffix = '_diff_etamax%d_nbins%d'%(etamax*10,nbins)

#### weak supervision
    model_weak = trainweak(samples,fracs,layersize, NB_EPOCH_weak, suffix, LEARNING_RATE)

    predict_weak = model_weak.predict_proba(X_test)
    area = getAUC(predict_weak, y_test)    
    return area

iqrs = []
medians = []
maxauc = []
for diff in diffs:
    toyfractions = [start_fraction, start_fraction + diff]
    print 'Current Fractions : ' + str(toyfractions)

    # Get Data
    samples,fracs,labels = getToys(toymeans,toystds,toyfractions)
    X_test, _, y_test = getToys(toymeans,toystds,[0.5, 0.5])
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)

    # Whiten the data
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

    # Perform Multiple runs
    aucs = []
    for i in range(nruns):
        print str(toyfractions) + ', Run ', i
        aucs.append(run(samples, fracs, X_test, y_test))

    # Compute Statistics
    medians.append(np.median(aucs))
    maxauc.append(max(aucs))
    q75, q25 = np.percentile(aucs, [75 ,25])
    iqr = q75-q25
    iqrs.append(iqr)

    print "<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>"
    print diff, np.median(aucs), iqr, aucs
    print "<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>"

SetupATLAS()

fig, ax1 = plt.subplots()
ax1.plot(diffs, iqrs,'b--',marker='o')
ax1.set_xlabel('$\Delta y$')
ax1.set_ylabel('$IQR(AUC)$',color='b')
#ax1.set_ylim([0.,0.1])
for tl in ax1.get_yticklabels():
    tl.set_color('b')

ax2 = ax1.twinx()
ax2.plot(diffs, maxauc,'r-.',marker='v',mfc='none')
ax2.plot(diffs, medians,'r--',marker='v')
ax2.set_ylabel('$<AUC>$',color='r')
ax2.set_ylim([0.83,0.9])
for tl in ax2.get_yticklabels():
    tl.set_color('r')

plt.tight_layout()
plt.savefig('paper_plots/fig2.png')
