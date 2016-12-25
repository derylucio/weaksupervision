import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from theano import tensor as T
from sklearn.cross_validation import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.regularizers import l2, l1

REGULARIZATION = 0 #5e-3
WEIGHT_REGULARIZATION = 0 #5e-3

def traincomplete(trainsamples,trainlabels,nb_epoch):
    X_train = np.concatenate( trainsamples )
    y_train = np.concatenate( trainlabels )
    
    model_complete = Sequential()
    model_complete.add( Dense(3, input_dim=(X_train.shape[1]), 
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
            
def loss_function(ytrue, ypred):
    # Assuming that ypred contains the same ratio replicated
    loss1 = K.sum(ypred)/ypred.shape[0] - K.sum(ytrue)/ypred.shape[0]
    constrib =  REGULARIZATION*K.std(ypred) 
    loss1 = K.square(loss1) - constrib
    
    loss2 = (1.0 - K.sum(ypred)/ypred.shape[0]) - K.sum(ytrue)/ypred.shape[0]
    loss2 = K.square(loss2) - constrib
    loss = K.switch(T.lt(loss1, loss2), loss1, loss2)
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
    checkpointer = ModelCheckpoint('weights'+suffix+'.h5', monitor='val_loss', save_best_only=True)
    earlystopper = EarlyStopping(monitor="val_loss", patience=2)
    model_weak.fit_generator(data_generator(listX_train, listf_train), trainsize, nb_epoch,
                             validation_data=data_generator(listX_val, listf_val), 
                             nb_val_samples=valsize, callbacks=[checkpointer])
    model_weak.load_weights('weights'+suffix+'.h5')
    return model_weak
