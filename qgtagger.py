from utils import getjetvar,evaluateModel
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from theano import tensor as T
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

run = 1
REGULARIZATION = 1e-4
NB_EPOCH = 10
features = ['n','w','eec2']
etamax = 2.1
nbins = 6
bins = np.linspace(-2.1,2.1,nbins)
scaler = StandardScaler()

def getSamples(features,etamax,bins):
    filename = 'data/20161103_16h20min.root'
    id0 = getjetvar('j0','id',filename)
    eta0 = getjetvar('j0','eta',filename)
    eta1 = getjetvar('j1','eta',filename)
    deta0 = getjetvar('j0','deta',filename)

    xx = [getjetvar('j0',var,filename) for var in features]
    X = np.vstack( [xx] ).T

    id0 = id0[(np.fabs(eta0)<etamax) & (np.fabs(eta1)<etamax)]
    deta0 = deta0[(np.fabs(eta0)<etamax) & (np.fabs(eta1)<etamax)]
    X = X[(np.fabs(eta0)<etamax) & (np.fabs(eta1)<etamax)]
    X = scaler.fit_transform(X)
    labels = id0<6

    samples = [X[(deta0>bins[i]) & (deta0<bins[i+1])] for i in range(len(bins)-1)]
    y = [labels[(deta0>bins[i]) & (deta0<bins[i+1])] for i in range(len(bins)-1)]
    output = [ [float(len(yy[yy==True]))/len(yy)]*len(yy) for yy in y]

    return samples,output,y
 
samples,fractions,labels = getSamples(features,etamax,bins)
print 'n bins',len(labels)
print 'sample sizes',[len(y) for y in labels]

trainsamples = []
trainlabels = []
trainfractions = []
testsamples = []
testlabels = []
for X,y,f in zip(samples,labels,fractions):
    X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(X, y, f, test_size=0.3)
    trainsamples.append(X_train)
    trainlabels.append(y_train)
    trainfractions.append(f_train)
    testsamples.append(X_test)
    testlabels.append(y_test)

### complete supervision
X_train = np.concatenate( trainsamples )
y_train = np.concatenate( trainlabels )

model_complete = Sequential()
model_complete.add( Dense(3, input_dim=(X_train.shape[1]), 
                          init='normal', activation='sigmoid') )
model_complete.add( Dense(1, init='normal', activation='sigmoid') )
model_complete.compile(loss='mean_squared_error', optimizer='sgd')
history = model_complete.fit(X_train, y_train, batch_size=128, nb_epoch=NB_EPOCH, 
                             validation_split=0.2)

#### weak supervision
def data_generator(samples, output):
    num_batches = len(samples)
    while 1:
        for i in xrange(num_batches):
            yield samples[i], output[i]

def loss_function(ytrue, ypred):
    # Assuming that ypred contains the same ratio replicated
    loss1 = K.sum(ypred)/ypred.shape[0] - K.sum(ytrue)/ypred.shape[0]
    constrib =  REGULARIZATION*K.std(ypred) 
    loss1 = K.square(loss1) - constrib
    
    loss2 = (1.0 - K.sum(ypred)/ypred.shape[0]) - K.sum(ytrue)/ypred.shape[0]
    loss2 = K.square(loss2) - constrib
    loss = K.switch(T.lt(loss1, loss2), loss1, loss2)
    return loss

listX_train = []
listX_val = []
listf_train = []
listf_val = []
for X,y,f in zip(trainsamples,trainlabels,trainfractions):
    X_train, X_val, y_train, y_val, f_train, f_val = train_test_split(X, y, f, test_size=0.2)
    listX_train.append(X_train)
    listf_train.append(f_train)
    listX_val.append(X_val)
    listf_val.append(f_val)

trainsize = sum([X.shape[0] for X in listX_train])
valsize = sum([X.shape[0] for X in listX_val])

model_weak = Sequential()
model_weak.add(Dense(3, input_dim=(len(features)), 
                     init='normal', activation='sigmoid') )
model_weak.add(Dense(1, init='normal', activation='sigmoid') )
model_weak.compile(loss=loss_function, optimizer=Adam(lr=0.001))
checkpointer = ModelCheckpoint('weights', monitor='val_loss', save_best_only=True)
model_weak.fit_generator(data_generator(listX_train, listf_train), trainsize, NB_EPOCH,
                         validation_data=data_generator(listX_val, listf_val), 
                         nb_val_samples=valsize, callbacks=[checkpointer])

###performance
import matplotlib.pyplot as plt
_, axarr = plt.subplots(3, 1)
axarr[0].set_xlabel('Gluon Jet efficiency')
axarr[0].set_ylabel('Quark Jet efficiency')
axarr[1].set_xlabel('probability')
axarr[2].set_xlabel('probability')

X_test = np.concatenate( testsamples )
y_test = np.concatenate( testlabels )
evaluateModel(axarr[1], axarr[0], model_complete, 'Complete Supervision', X_test, y_test)
evaluateModel(axarr[2], axarr[0], model_weak, 'Weak Supervision', X_test, y_test)

plt.savefig('etamax%d_nbins%d_run%d'%(etamax*10,nbins,run))
