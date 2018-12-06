# -*- coding: utf-8 -*-

import numpy as np
import Dataset as ds
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pickle
import time
import matplotlib.pyplot as plt

np.random.seed(42)
nb_frame=60

def maj_vote(res):
    res=res
    vote=[]
    for j in res:
        li=[]
        for i in range(nb_frame):
            li.append(j[i])
        vote.append(max(li,key=li.count))
    return vote

def acc_seq(y_pred,Y):
	res=[]
	for i in range(0,len(Y)):
		temp=[]
		for j in range(0,nb_frame):
			temp.append(y_pred[i*nb_frame+j])
		res.append(temp)
	vote_test=maj_vote(res)
	print(accuracy_score(vote_test,Y))#,np.argmax(Y, axis=1)))



experiment_name = 'Classify_mvts'
train = ds.DataSet('../data2/',720, onehot=False, load=True)

print(train.label.shape)
print(train.data.shape)


X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train.data, train.label, train_size=0.75, test_size=0.25)
x_train,x_test,y_train,y_test=[],[],[],[]


for i in X_train:
    for j in i:
        x_train.append(j)
for i in X_test:
    for j in i:
        x_test.append(j)
for i in Y_train:
<<<<<<< HEAD
    for j in range(0,nb_frame):
        y_train.append(i)
for i in Y_test:
    for j in range(0,nb_frame):
        y_test.append(i)

print(np.array(y_train).shape)
rf=RandomForestClassifier(n_estimators=50,n_jobs=-1,criterion="gini")
rf.fit(x_train,y_train)
with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)
print("trained model!")
#print(rf.score(x_train,y_train))

y_pred_train=rf.predict(x_train)
y_pred_test=rf.predict(x_test)
#print(confusion_matrix(y_train,y_pred))

Y_test=[]

for i in range(0,len(y_test),nb_frame):
    Y_test.append(y_test[i])

#time_n = time.time()

acc_seq(y_pred_test,Y_test)
fi=rf.feature_importances_
print(fi)
def get_indices(rf,n):
	fi=np.asarray(rf.feature_importances_)
	li=fi.argsort()[::-1][:n]
	print(li)
	with open('n__features_indices.pkl', 'wb') as f:
		pickle.dump(li,f)
		
get_indices(rf,100)
#plt.plot(fi)
#plt.show()
#for feat in rf.feature_importances_:
#    print(feat)
#print("time between frames:", (time.time()-time_n)/240)
