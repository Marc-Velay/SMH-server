#import tensorflow as tf
import numpy as np
import Dataset as ds


def get_dict(database):
	xs,ys = database.NextTrainingBatch()
	return {x:xs,y_desired:ys}


experiment_name = 'Classify_mvts'
train = ds.DataSet('../data2/',720)

X_train, X_test, y_train, y_test = model_selection.train_test_split(train.data, train.label, train_size=0.7, test_size=0.3)
