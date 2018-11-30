import numpy as np

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, LSTM, TimeDistributed
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Nadam


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


def create_LSTM2(xshape, yshape, batchSize):

	print(yshape)
	x = Input(shape=(xshape[1], xshape[2]))

	cell1 = LSTM(500, return_sequences=True,batch_input_shape=(batchSize, xshape[1], xshape[2]))(x)
	cell2 = LSTM(500, return_sequences=True)(cell1)
	cell3 = LSTM(500)(cell2)
	d1 = Dense(50, activation='sigmoid')(cell3) #attention layer
	#6 is the number of classes
	intents  = Dense(6, activation='sigmoid', name='mvt_class')(d1)

	#intents = Dense(6, activation='sigmoid', name='mvt_class')(d1)

	model = Model(inputs=x, outputs=intents)
	opt = Nadam(lr=0.00001 )
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	print(model.summary())
	return model,opt


def create_LSTM3(xshape, yshape, batchSize):

	print(yshape)
	x = Input(shape=(xshape[1], xshape[2]))

	cell1 = LSTM(500, return_sequences=True,batch_input_shape=(batchSize, xshape[1], xshape[2]))(x)
	cell2 = LSTM(250, return_sequences=True)(cell1)
	cell3 = LSTM(200, return_sequences=True)(cell2)
	cell4 = LSTM(100, return_sequences=True)(cell3)
	cell5 = LSTM(50)(cell4)
	d1 = Dense(50, activation='sigmoid')(cell5) #attention layer
	#6 is the number of classes
	intents  = Dense(6, activation='sigmoid', name='mvt_class')(d1)

	#intents = Dense(6, activation='sigmoid', name='mvt_class')(d1)

	model = Model(inputs=x, outputs=intents)
	opt = Nadam(lr=0.00001 )
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	print(model.summary())
	return model,opt
