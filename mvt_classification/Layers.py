import tensorflow as tf
import numpy as np


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
