from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib
import random
from keras import backend as K
from theano import tensor as T

matplotlib.use('pdf')
# scan the number of examples
# understand why current example does worse than random.
FILE_NAME = "Sanity-Check-flip-1024-std-0-0001-0499-0501"
NUM_TESTS = 20
SAMPLES_PER_EPOCH = 512000
NB_EPOCH = 5
NB_VAL_SAMPLES = 12800
BATCH_SIZE = 4096
NUM_FEATURES = 3
FRACTIONS = [0.499, 0.501]  #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NUM_TEST_SAMPLES = 10000
STD_IMP = 1e-4
scaler = StandardScaler()

def generateTrainSamples(totalSamples):
	num_batches = totalSamples/BATCH_SIZE
	samples = []
	output = []
	ty = []
	for i in xrange(num_batches):
		ind = i % len(FRACTIONS)
		num_signal = int(FRACTIONS[ind]*BATCH_SIZE)
		num_bckg = BATCH_SIZE - num_signal
		signal = np.stack([np.random.normal(13,5,num_signal),
						np.random.normal(0.1,0.3,num_signal), 
						np.random.normal(0.24,0.05,num_signal)])
		bckg = np.stack([
		        np.random.normal(19,5, num_bckg),
		        np.random.normal(0.15,0.3,num_bckg),
		        np.random.normal(0.29,0.04,num_bckg)])
		signal = signal.T
		bckg = bckg.T
		X = np.concatenate((signal,bckg))
		X = scaler.fit_transform(X)
		permutation = np.random.permutation(len(X))
		X = X[permutation]
		samples.append(X)

		output.append([FRACTIONS[ind]]*BATCH_SIZE)
		y = np.concatenate( (np.ones(signal.shape[0]),np.zeros(bckg.shape[0])) )
		y = y[permutation]
		ty.append(y)
	return samples, output, ty

def generateTestSamples(num_signal, num_bckg):
	signal = np.stack([np.random.normal(13,5,num_signal),
					np.random.normal(0.1,0.3,num_signal), 
					np.random.normal(0.24,0.05,num_signal)])
	bckg = np.stack([
	        np.random.normal(19,5, num_bckg),
	        np.random.normal(0.15,0.3,num_bckg),
	        np.random.normal(0.29,0.04,num_bckg)])
	signal = signal.T
	bckg = bckg.T
	X = np.concatenate((signal,bckg))
	X = scaler.fit_transform(X)
	y = np.concatenate( (np.ones(signal.shape[0]),np.zeros(bckg.shape[0])) )
	return X, y

def data_generator(samples, output):
	num_batches = len(samples)
	while 1:
		for i in xrange(num_batches):
			yield samples[i], output[i]

def custom_objective_nostd(ytrue, ypred):
	# Assuming that ypred contains the same ratio replicated
	loss1 = K.sum(ypred)/BATCH_SIZE - K.sum(ytrue)/BATCH_SIZE
	loss1 = K.square(loss1)

	loss2 = (1.0 - K.sum(ypred)/BATCH_SIZE) - K.sum(ytrue)/BATCH_SIZE
	loss2 = K.square(loss2)
	loss = K.switch(T.lt(loss1, loss2), loss1, loss2)
	return loss

def custom_objective_std(ytrue, ypred):
	# Assuming that ypred contains the same ratio replicated
	loss1 = K.sum(ypred)/BATCH_SIZE - K.sum(ytrue)/BATCH_SIZE
	constrib =  STD_IMP*K.std(ypred) 
	loss1 = K.square(loss1) - constrib

	loss2 = (1.0 - K.sum(ypred)/BATCH_SIZE) - K.sum(ytrue)/BATCH_SIZE
	loss2 = K.square(loss2) - constrib
	loss = K.switch(T.lt(loss1, loss2), loss1, loss2)
	return loss

def evaluateModel(model, x_test, y_test):
	predict_proba = model.predict_proba(x_test)
	fpr,tpr,thres = roc_curve(y_test, predict_proba)
	area =  auc(fpr, tpr)
	if area < 0.5:
		fpr,tpr,thres = roc_curve(y_test, 1.0 - predict_proba)
		area =  auc(fpr, tpr)
	return area

def trainModel(train_samples, train_output, val_samples, val_output, loss_function, savefilename):
	model = Sequential()
	model.add(Dense(3, input_dim=(NUM_FEATURES), init='he_normal', activation='sigmoid') )
	# model.add(Dense(10, input_dim=(NUM_FEATURES), init='normal', activation='sigmoid') )
	# model.add(Dense(10, input_dim=(NUM_FEATURES), init='normal', activation='sigmoid') )
	# model.add(Dense(5, input_dim=(NUM_FEATURES), init='normal', activation='sigmoid') )
	model.add(Dense(1, init='he_normal', activation='sigmoid') )
	# optimizer = Adam(lr=0.1)
	# model.compile(loss=loss_function, optimizer=optimizer)
	model.compile(loss=loss_function, optimizer='Adam')
	checkpointer = ModelCheckpoint(savefilename, monitor='val_loss', save_best_only=True)
	model.fit_generator(data_generator(train_samples, train_output), SAMPLES_PER_EPOCH, NB_EPOCH, \
	 validation_data=data_generator(val_samples, val_output), nb_val_samples=NB_VAL_SAMPLES, callbacks=[checkpointer])
	return model

train_samples, train_output, train_ty = generateTrainSamples(SAMPLES_PER_EPOCH)
val_samples, val_output, val_ty  = generateTrainSamples(NB_VAL_SAMPLES)
x_test, y_test = generateTestSamples(NUM_TEST_SAMPLES,NUM_TEST_SAMPLES)

train_samples = np.array(train_samples)
train_ty = np.array(train_ty)
print 'unrolling batches of data into single vector of samples'
supervised_samples = np.reshape(train_samples, (train_samples.shape[1]*train_samples.shape[0], train_samples.shape[2]))
supervised_output = np.reshape(train_ty, (SAMPLES_PER_EPOCH, 1))

data = np.zeros((3, NUM_TESTS))
for i in xrange(NUM_TESTS):
	print 'epoch ', i
	print 'Training Model No STD augmentation'
	model_nostd = trainModel(train_samples, train_output, val_samples, val_output, custom_objective_nostd, 'weights_nostd')
	data[0][i] = evaluateModel(model_nostd, x_test, y_test)

	print 'Training Model STD augmentation'
	model_std = trainModel(train_samples, train_output, val_samples, val_output, custom_objective_std, 'weights_std')
	data[1][i] = evaluateModel(model_std, x_test, y_test)

	print 'Training complete supervision Model'
	model_complete = Sequential()
	model_complete.add( Dense(3, input_dim=(supervised_samples.shape[1]), init='he_normal', activation='sigmoid') )
	# model_complete.add( Dense(10, input_dim=(supervised_samples.shape[1]), init='normal', activation='sigmoid') )
	# model_complete.add( Dense(10, input_dim=(supervised_samples.shape[1]), init='normal', activation='sigmoid') )
	# model_complete.add( Dense(5, input_dim=(supervised_samples.shape[1]), init='normal', activation='sigmoid') )
	model_complete.add( Dense(1, init='he_normal', activation='sigmoid') )
	# optimizer = Adam(lr=0.1)
	# model_complete.compile(loss='mean_squared_error', optimizer=optimizer)
	model_complete.compile(loss='mean_squared_error', optimizer='Adam')
	history = model_complete.fit(supervised_samples, supervised_output, batch_size=128, nb_epoch=NB_EPOCH, validation_split=0.2)
	data[2][i] = evaluateModel(model_complete, x_test, y_test)

plt.plot(data[0, :], label='Weak Supervision-No_STD')
plt.plot(data[1, :], label='Weak Supervision-STD')
plt.plot(data[2, :], label='Complete Supervision')
plt.xlabel('Run')
plt.ylabel('Area Under Curve')
lgd = plt.legend(loc='upper left', bbox_to_anchor=(1, 1.25))
plt.savefig(FILE_NAME,  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close()

# same sample, vary learning rate/optimizer
# multiple times on the same sample 
# using optimal learn rate/ optimizer - running multiple new samples

