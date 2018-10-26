#import tensorflow as tf
import numpy as np
import Dataset as ds


def get_dict(database):
	xs,ys = database.NextTrainingBatch()
	return {x:xs,y_desired:ys}


experiment_name = 'Classify_mvts'
train = ds.DataSet('../data2/',720)
