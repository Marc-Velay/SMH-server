#import tensorflow as tf
import numpy as np
import Dataset as ds
from Layers import *
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix
import math
import matplotlib.pyplot as plt


def get_dict(database):
	xs,ys = database.NextTrainingBatch()
	return {x:xs,y_desired:ys}

experiment_name = 'Classify_mvts'
load_data = True
train = ds.DataSet('../data2/',720, load=load_data)

batchSize = 10
batchSizetest = 1

X_train, X_test, y_train, y_test = model_selection.train_test_split(train.data, train.label, train_size=0.75, test_size=0.25)

print('define lstm model')
#model, opt = get_lstm(X_train.shape, y_train.shape, y2_train.shape, BATCH_SIZE)
model, opt = create_LSTM(X_train.shape, y_train.shape, batchSize)

# define the checkpoint
filepath="weights/weights-lstm.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#history = model.fit(X_train[:batchSize*math.floor(X_train.shape[0]/batchSize)], y_train[:batchSize*math.floor(X_train.shape[0]/batchSize)],
#                    epochs=20, batch_size=batchSize, callbacks=callbacks_list, validation_split=0.2)
model.load_weights(filepath)

predictions = model.predict(X_test[:batchSizetest*math.floor(X_test.shape[0]/batchSizetest)], batch_size=batchSizetest)
print("confusion matrix: ")
conf = confusion_matrix([np.argmax(y) for y in y_test[:batchSizetest*math.floor(X_test.shape[0]/batchSizetest)]], [np.argmax(y) for y in predictions])
print(conf)
accu_score = accuracy_score([np.argmax(y) for y in y_test[:batchSizetest*math.floor(X_test.shape[0]/batchSizetest)]], [np.argmax(y) for y in predictions])
print("LSTM acc: ", accu_score)

"""
plt.figure(1)
plt.plot(history.history['loss'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()
"""
