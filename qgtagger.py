from utils import evaluateModel
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from dataprovider import getSamples,getToys
from models import traincomplete,trainweak

nruns = 10
layersize = 30
NB_EPOCH = 40
NB_EPOCH_weak = 25
LEARNING_RATE = 9e-3
features = ['n','w','eec2']
etamax = 2.1
nbins = 12
bins = np.linspace(-2.1,2.1,nbins+1)
usetoys = True
toymeans = [(18,26),(0.06,0.09),(0.23,0.28)] 
toystds  = [(7,8),  (0.04,0.04),(0.05,0.04)]
toyfractions =  [0.24, 0.25, 0.25, 0.26, 0.27, 0.29, 0.31, 0.33, 0.37, 0.39, 0.44]
scaler = StandardScaler()

def run(nrun, trainsamples, trainlabels, trainfractions, X_test, y_test): 
    suffix = '_tagger_etamax%d_nbins%d'%(etamax*10,nbins)
    
    ### weak supervision
    model_weak = trainweak(trainsamples,trainfractions,layersize, NB_EPOCH_weak ,suffix, LEARNING_RATE)
    auc_weak = evaluateModel(plt, model_weak.predict_proba(X_test), y_test, 'WeakSupervision')

    ### complete supervision
    model_complete = traincomplete(trainsamples, trainlabels,NB_EPOCH)
    auc_sup = evaluateModel(plt, model_complete.predict_proba(X_test), y_test, 'CompleteSupervision')

    return auc_sup, auc_weak

# Get the data
if usetoys:
    samples,fractions,labels = getToys(toymeans,toystds,toyfractions)
else:
    samples,fractions,labels = getSamples(features,etamax,bins)

X_test, _, y_test = getToys(toymeans,toystds,[0.5, 0.5])
X_test = np.concatenate(X_test)
y_test = np.concatenate(y_test)

# Whiten Data
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

all_aucs = []
for runnum in range(nruns):

    auc_sup,auc_weak = run(runnum, samples, labels, fractions, X_test, y_test)
    all_aucs.append([auc_sup, auc_weak])
    if runnum == 0:
        num_features = len(toymeans) if usetoys else len(features)
        for i in range(num_features):
            feature = X_test[:, i]
            evaluateModel(plt, feature, y_test, 'Feature ' + str(i + 1))
        plt.xlabel('True Positive Rate')
        plt.ylabel('False Positive Rate')
        plt.legend(loc='lower right', title='Weak Supervision v Complete Supervision', frameon=True)
        plt.savefig('toy_plots/weak_vs_complete')
        plt.close()
plt.close()

all_aucs = np.array(all_aucs)
plt.xlabel("run")
plt.ylabel("auc")
plt.ylim([0, 1])
plt.plot(all_aucs[:, 0], label='Complete Supervision')
plt.plot(all_aucs[:, 1], label='Weak Supervision')
plt.legend(loc='lower right', frameon=True)
plt.savefig('toy_plots/weavk_vs_complete_stability')

