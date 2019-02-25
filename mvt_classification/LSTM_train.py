#import tensorflow as tf
import numpy as np
import Dataset as ds
from Layers import *
import keras_callbacks
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix
import math
import matplotlib.pyplot as plt

#np.random.seed(42)

def get_dict(database):
	xs,ys = database.NextTrainingBatch()
	return {x:xs,y_desired:ys}

experiment_name = 'Classify_mvts'

load_data = True
TRAIN = True
batchSize = 50
batchSizetest = 1
train = ds.DataSet('../data3/',841, load=load_data, onehot=True, batchSize=batchSize)

print("x shape", train.data.shape)
X_train, X_test, y_train, y_test = model_selection.train_test_split(train.data, train.label, train_size=0.999, test_size=0.001)

print('define lstm model')
model, opt = create_LSTM4(X_train.shape, y_train.shape, batchSize)
print("X shape", X_train.shape, "y shape", y_train.shape)
# define the checkpoint
filepath="weights/weights-lstm6.hdf5"

if TRAIN:
	histories = keras_callbacks.Histories()
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
	callbacks_list = [checkpoint, histories]
	history = model.fit(X_train[:batchSize*math.floor(X_train.shape[0]/batchSize)], y_train[:batchSize*math.floor(X_train.shape[0]/batchSize)],
	                    epochs=25, batch_size=batchSize, callbacks=callbacks_list, validation_split=0.2)
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

#*needs to be shifted out of if
model.load_weights(filepath)

predictions = model.predict(X_test[:batchSizetest*math.floor(X_test.shape[0]/batchSizetest)], batch_size=batchSizetest)
print("confusion matrix: ")
conf = confusion_matrix([np.argmax(y) for y in y_test[:batchSizetest*math.floor(X_test.shape[0]/batchSizetest)]], [np.argmax(y) for y in predictions])
print(conf)
accu_score = accuracy_score([np.argmax(y) for y in y_test[:batchSizetest*math.floor(X_test.shape[0]/batchSizetest)]], [np.argmax(y) for y in predictions])
print("LSTM acc: ", accu_score)
#*
