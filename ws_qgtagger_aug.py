from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import matplotlib
import random
from keras import backend as K
from theano import tensor as T

matplotlib.use('pdf')
# scan the number of examples
# understand why current example does worse than random.
run = "3"
SAMPLES_PER_EPOCH = 512000
NB_EPOCH = 10
NB_VAL_SAMPLES = 12800
BATCH_SIZE = 4096
NUM_FEATURES = 3
FRACTIONS = [0.2, 0.3] #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NUM_TEST_SAMPLES = 10000
STD_IMP = 1e-3
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
		samples.append(X)
		output.append([FRACTIONS[ind]]*BATCH_SIZE)
		y = np.concatenate( (np.ones(signal.shape[0]),np.zeros(bckg.shape[0])) )
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

def custom_objective(ytrue, ypred):
	# Assuming that ypred contains the same ratio replicated
	loss1 = K.sum(ypred)/BATCH_SIZE - K.sum(ytrue)/BATCH_SIZE
	constrib =  STD_IMP*K.std(ypred) 
	loss1 = K.square(loss1) - constrib

	loss2 = (1.0 - K.sum(ypred)/BATCH_SIZE) - K.sum(ytrue)/BATCH_SIZE
	loss2 = K.square(loss2) - constrib
	loss = K.switch(T.lt(loss1, loss2), loss1, loss2)
	return loss

def evaluateModel(plt_ax, hist_ax,  model, label, x_test, y_test):
	predict_proba = model.predict_proba(x_test)
	fpr,tpr,thres = roc_curve(y_test, predict_proba)	
	hist_ax.hist(predict_proba[y_test == 1], histtype='step', normed=True, label = label  + ' signal')
	hist_ax.hist(predict_proba[y_test == 0], histtype='step',  normed=True, label = label + ' background')
	area =  auc(fpr, tpr)
	print label, area
	plt_ax.plot(fpr, tpr, label=label)

print 'Getting Training Samples'
train_samples, train_output, train_ty = generateTrainSamples(SAMPLES_PER_EPOCH)
val_samples, val_output, val_ty  = generateTrainSamples(NB_VAL_SAMPLES)


print 'Training Model without STD augmentation'
model_nstd = Sequential()
model_nstd.add(Dense(3, input_dim=(NUM_FEATURES), init='normal', activation='sigmoid') )
model_nstd.add(Dense(1, init='normal', activation='sigmoid') )
model_nstd.compile(loss=custom_objective_nostd, optimizer='Adam')
checkpointer = ModelCheckpoint('weights', monitor='val_loss', save_best_only=True)
history = model_nstd.fit_generator(data_generator(train_samples, train_output), SAMPLES_PER_EPOCH, NB_EPOCH, validation_data=data_generator(val_samples, val_output), nb_val_samples=NB_VAL_SAMPLES, callbacks=[checkpointer])

print 'Training Model with STD augmentation'
model = Sequential()
model.add(Dense(3, input_dim=(NUM_FEATURES), init='normal', activation='sigmoid') )
model.add(Dense(1, init='normal', activation='sigmoid') )
model.compile(loss=custom_objective, optimizer='Adam')
checkpointer = ModelCheckpoint('weights', monitor='val_loss', save_best_only=True)
history = model.fit_generator(data_generator(train_samples, train_output), SAMPLES_PER_EPOCH, NB_EPOCH, validation_data=data_generator(val_samples, val_output), nb_val_samples=NB_VAL_SAMPLES, callbacks=[checkpointer])

_, axarr = plt.subplots(3, 1)
axarr[0].set_xlabel('True Positive')
axarr[0].set_ylabel('False Positve')
axarr[1].set_xlabel('Bin')
axarr[1].set_ylabel('Fraction')
axarr[2].set_xlabel('Bin')
axarr[2].set_ylabel('Fraction')

x_test, y_test = generateTestSamples(NUM_TEST_SAMPLES,NUM_TEST_SAMPLES)
print 'Evaluating Weak Supervision Model - NO_STD'
evaluateModel(axarr[0], axarr[1], model_nstd, 'Weak Supervision - No STD', x_test, y_test)



print 'Evaluating Complete Supervision Model'
train_samples = np.array(train_samples)
train_ty = np.array(train_ty)
supervised_samples = np.reshape(train_samples, (train_samples.shape[1]*train_samples.shape[0], train_samples.shape[2]))
print supervised_samples.shape
print train_ty.shape
supervised_output = np.reshape(train_ty, (SAMPLES_PER_EPOCH, 1))
print supervised_output.shape

model2 = Sequential()
model2.add( Dense(3, input_dim=(supervised_samples.shape[1]), init='normal', activation='sigmoid') )
model2.add( Dense(1, init='normal', activation='sigmoid') )
model2.compile(loss='mean_squared_error', optimizer='sgd')

history = model2.fit(supervised_samples, supervised_output, batch_size=128, nb_epoch=NB_EPOCH, validation_split=0.2)
evaluateModel(axarr[0], axarr[2], model2, 'Complete Supervision', x_test, y_test)


random_val = np.random.uniform(size=(len(y_test)))
print len(random_val)
fpr,tpr,thres = roc_curve(y_test, random_val)	
area =  auc(fpr, tpr)
print area


lgd1 = axarr[0].legend(loc='upper left', bbox_to_anchor=(1, 1.25))         
lgd2 = axarr[1].legend(loc='upper left', bbox_to_anchor=(1, 1.25))
lgd3 = axarr[2].legend(loc='upper left', bbox_to_anchor=(1, 1.25))         

plt.savefig('Weak-Supervision-Study-0304-NoSTD_FlippedLoss_' + run,bbox_extra_artists=(lgd1, lgd2, lgd3,), bbox_inches='tight')
plt.close()

_, axarr = plt.subplots(3, 1)
axarr[0].set_xlabel('True Positive')
axarr[0].set_ylabel('False Positve')
axarr[1].set_xlabel('Bin')
axarr[1].set_ylabel('Fraction')
axarr[2].set_xlabel('Bin')
axarr[2].set_ylabel('Fraction')


print 'Evaluating Weak Supervision Model - STD'
evaluateModel(axarr[0], axarr[1], model, 'Weak Supervision - STD', x_test, y_test)
evaluateModel(axarr[0], axarr[2], model2, 'Complete Supervision', x_test, y_test)

lgd1 = axarr[0].legend(loc='upper left', bbox_to_anchor=(1, 1.25))         
lgd2 = axarr[1].legend(loc='upper left', bbox_to_anchor=(1, 1.25))
lgd3 = axarr[2].legend(loc='upper left', bbox_to_anchor=(1, 1.25))    
plt.savefig('Weak-Supervision-Study-0304-STD_FlippedLoss_' + run ,bbox_extra_artists=(lgd1, lgd2, lgd3,), bbox_inches='tight')
plt.close()

