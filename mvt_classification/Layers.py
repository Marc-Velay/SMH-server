import tensorflow as tf
import numpy as np

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, LSTM, TimeDistributed
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Nadam

def variable_summaries(var, name):
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar( name + '/mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
		tf.summary.scalar( name + '/sttdev' , stddev)
		tf.summary.scalar( name + '/max' , tf.reduce_max(var))
		tf.summary.scalar( name + '/min' , tf.reduce_min(var))
		tf.summary.histogram(name, var)

def fc(tensor, output_dim, name, act=tf.nn.relu):
	with tf.name_scope(name):
		input_dim = tensor.get_shape()[1].value
		Winit = tf.truncated_normal([input_dim, output_dim], stddev=0.1)
		W = tf.Variable(Winit)
		print (name,'input  ',tensor)
		print (name,'W  ',W.get_shape())
		variable_summaries(W, name + '/W')
		Binit = tf.constant(0.0, shape=[output_dim])
		B = tf.Variable(Binit)
		variable_summaries(B, name + '/B')
		tensor = tf.matmul(tensor, W) + B
		tensor = act(tensor)
	return tensor


def dense(tensor, outDim, name):
	with tf.name_scope(name):
		inDim = tensor.get_shape()[1].value
		Winit = tf.Variable(tf.truncated_normal([inDim, outDim], mean=0, stddev=1 / np.sqrt(inDim)), name='weights1')
		W = tf.Variable(Winit)
		print (name,'input  ',tensor)
		print (name,'W  ',W.get_shape())
		variable_summaries(W, name + '/W')
		Binit = tf.Variable(tf.truncated_normal([outDim],mean=0, stddev=1 / np.sqrt(inDim)), name='bias1')
		B = tf.Variable(Binit)
		variable_summaries(B, name + '/B')
		tensor = tf.matmul(tensor, W)+B
		tensor = tf.nn.tanh(tensor, name='activationLayer1')
	return tensor

def create_LSTM(xshape, yshape, batchSize):

	print(yshape)
	x = Input(shape=(xshape[1], xshape[2]))

	cell1 = LSTM(2000, return_sequences=True,batch_input_shape=(batchSize, xshape[1], xshape[2]))(x)
	cell2 = LSTM(500)(cell1)
	d1 = Dense(50, activation='sigmoid')(cell2) #attention layer
	#6 is the number of classes
	intents  = Dense(6, activation='sigmoid', name='mvt_class')(d1)

	#intents = Dense(6, activation='sigmoid', name='mvt_class')(d1)

	model = Model(inputs=x, outputs=intents)
	opt = Nadam(lr=0.00001 )
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

	return model,opt
