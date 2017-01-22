import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from theano import tensor as T
from sklearn.cross_validation import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD
from keras.regularizers import l2, l1
import os

WEIGHT_REGULARIZATION = 0
REGULARIZATION = 0

def traincomplete(trainsamples,trainlabels,nb_epoch):
    X_train = np.concatenate( trainsamples )
    y_train = np.concatenate( trainlabels )
    signal_frac = sum(y_train)*1.0/len(y_train)
    
    model_complete = Sequential()
    model_complete.add( Dense(3, input_dim=(X_train.shape[1]), 
                              init='normal', activation='sigmoid') )
    model_complete.add( Dense(1, init='normal', activation='sigmoid') )
    model_complete.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01))
    save_file_name = 'complete_weights.h5'
    if os.path.exists(save_file_name):
        print 'Complete Supervision Weight File Exists. Replacing ...'
        os.remove(save_file_name)
    checkpointer = ModelCheckpoint(save_file_name, monitor='val_loss', save_best_only=True)
    history = model_complete.fit(X_train, y_train, batch_size=128, nb_epoch=nb_epoch, 
                                 validation_split=0.2, callbacks=[checkpointer])
    model_complete.load_weights(save_file_name)
    return model_complete

def data_generator(samples, output):
    num_batches = len(samples)
    print 'Num Batches', num_batches
    while 1:
        for i in xrange(num_batches):
            yield samples[i], output[i]
            
def loss_function(ytrue, ypred):
    # Assuming that ypred contains the same ratio replicated
    loss = K.sum(ypred)/ypred.shape[0] - K.sum(ytrue)/ypred.shape[0]
    constrib =  REGULARIZATION*K.std(ypred) 
    loss = K.square(loss) - constrib
    return loss

def trainweak(trainsamples,trainfractions,layersize,nb_epoch,suffix, learning_rate):
    listX_train = []
    listX_val = []
    listf_train = []
    listf_val = []
    for X,f in zip(trainsamples,trainfractions):
        X_train, X_val, f_train, f_val = train_test_split(X, f, test_size=0.2)
        listX_train.append(X_train)
        listf_train.append(f_train)
        listX_val.append(X_val)
        listf_val.append(f_val)
    
    trainsize = sum([X.shape[0] for X in listX_train])
    valsize = sum([X.shape[0] for X in listX_val])
    
    model_weak = Sequential()
    model_weak.add(Dense(layersize, input_dim=(trainsamples[0].shape[1]), 
                         init='normal', activation='sigmoid', W_regularizer=l2(WEIGHT_REGULARIZATION)) )
    model_weak.add(Dense(1, init='normal', activation='sigmoid') )
    model_weak.compile(loss=loss_function, optimizer=Adam(lr=learning_rate))
    save_file_name = 'weights'+suffix+'.h5'
    if os.path.exists(save_file_name):
        print 'Weak Supervision Weight File Exists. Replacing ...'
        os.remove(save_file_name)
    checkpointer = ModelCheckpoint(save_file_name, monitor='val_loss', save_best_only=True)
    earlystopper = EarlyStopping(monitor="val_loss", patience=2)
    model_weak.save(save_file_name)
    model_weak.fit_generator(data_generator(listX_train, listf_train), trainsize, nb_epoch,
                             validation_data=data_generator(listX_val, listf_val), 
                             nb_val_samples=valsize, callbacks=[checkpointer])
    model_weak.load_weights(save_file_name)
    return model_weak
