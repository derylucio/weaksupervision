from utils import evaluateModel,SetupATLAS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from dataprovider import getToys
from models import traincomplete,trainweak

layersize = 30
NB_EPOCH = 40
NB_EPOCH_weak = 25
LEARNING_RATE = 9e-3
toymeans = [(18,26),(0.06,0.09),(0.23,0.28),(26,18),(0.09,0.06)] 
toystds  = [(7,8),  (0.04,0.04),(0.05,0.04),(7,8),  (0.04,0.04)]
toyfractions = np.linspace(0.2,0.4,9)
scaler = StandardScaler()

def run(trainsamples, trainlabels, trainfractions, X_test, y_test): 

    ### weak supervision
    model_weak = trainweak(trainsamples,trainfractions,layersize, NB_EPOCH_weak,LEARNING_RATE)
    evaluateModel(plt, model_weak.predict_proba(X_test), y_test, 'Weakly supervised NN')

    ### complete supervision
    model_complete = traincomplete(trainsamples, trainlabels,NB_EPOCH)
    evaluateModel(plt, model_complete.predict_proba(X_test), y_test, 'Fully supervised NN')

# Get the data
samples,fractions,labels = getToys(toymeans,toystds,toyfractions)

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

#SetupATLAS()

run(samples, labels, fractions, X_test, y_test)
num_features = len(toymeans)
for i in range(num_features):
    feature = X_test[:, i]
    evaluateModel(plt, feature, y_test, 'Feature ' + str(i + 1))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right', frameon=False)
plt.savefig('paper_plots/fig4.png')
plt.close()
