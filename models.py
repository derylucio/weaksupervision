import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import backend as K
from theano import tensor as T
from sklearn.cross_validation import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.regularizers import l2, l1

def traincomplete(trainsamples,trainlabels,nb_epoch):
    X_train = np.concatenate( trainsamples )
    y_train = np.concatenate( trainlabels )
    
    model_complete = Sequential()
    model_complete.add( Dense(64, input_dim=(X_train.shape[1]), 
                              init='uniform', activation='relu') )
    model_complete.add(Dropout(0.5))
    model_complete.add(Dense(64,activation='relu'))
    model_complete.add(Dropout(0.5))
    model_complete.add( Dense(1, activation='sigmoid') )
    model_complete.compile(loss='mean_squared_error', optimizer='sgd')
    history = model_complete.fit(X_train, y_train, batch_size=128, nb_epoch=nb_epoch, 
                                 validation_split=0.2)
    return model_complete

def trainqgcomplete(trainsamples,trainlabels,nb_epoch,layersize=10):
    X_train = np.concatenate( trainsamples )
    y_train = np.concatenate( trainlabels )
    
    model_complete = Sequential()
    model_complete.add( Dense(layersize, input_dim=(X_train.shape[1]), 
                              init='normal', activation='sigmoid') )
    model_complete.add( Dense(1, init='normal', activation='sigmoid') )
    model_complete.compile(loss='mean_squared_error', optimizer='sgd')
    history = model_complete.fit(X_train, y_train, batch_size=128, nb_epoch=nb_epoch, 
                                 validation_split=0.2)
    return model_complete

def data_generator(samples, output):
    num_batches = len(samples)
    while 1:
        for i in xrange(num_batches):
            yield samples[i], output[i]

def getweak(inputsize,layersize,l2reg=0,sdreg=0):
    def loss_function(ytrue, ypred):
        # Assuming that ypred contains the same ratio replicated
        loss1 = K.sum(ypred)/ypred.shape[0] - K.sum(ytrue)/ypred.shape[0]
        constrib = sdreg*K.std(ypred) 
        loss1 = K.square(loss1) - constrib
        
        loss2 = (1.0 - K.sum(ypred)/ypred.shape[0]) - K.sum(ytrue)/ypred.shape[0]
        loss2 = K.square(loss2) - constrib
        loss = K.switch(T.lt(loss1, loss2), loss1, loss2)
        return loss
    
    model_weak = Sequential()
    model_weak.add(Dense(64, input_dim=(inputsize), 
                         init='uniform', activation='relu', 
                         W_regularizer=l2(l2reg)) )
    model_weak.add(Dropout(0.5))
    model_weak.add(Dense(64, activation='relu', W_regularizer=l2(l2reg)) )
    model_weak.add(Dropout(0.5))
    model_weak.add(Dense(1, activation='sigmoid'
                         ))
    model_weak.compile(loss=loss_function, optimizer=Adam(lr=0.001))
    
    return model_weak

def trainweak(trainsamples,trainfractions,layersize,nb_epoch,l2reg=0,sdreg=0,suffix=''):
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

    inputsize = trainsamples[0].shape[1]
    model_weak = getweak(inputsize,layersize,l2reg,sdreg)
#    checkpointer = ModelCheckpoint('weights'+suffix+'.h5', monitor='val_loss', 
#                                   save_best_only=True)
    model_weak.fit_generator(data_generator(listX_train, listf_train), trainsize, 
                             nb_epoch,
                             validation_data=data_generator(listX_val, listf_val), 
                             nb_val_samples=valsize
                             #, callbacks=[checkpointer]
                             )

    return model_weak
