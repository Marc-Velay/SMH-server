# -*- coding: utf-8 -*-

import numpy as np
import Dataset as ds
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pickle
import time

np.random.seed(42)


def maj_vote(res):
    #res=res
    vote=[]
    for i in res:
        idx = list(np.argmax(i, axis=1))
        vote.append(max(idx,key=idx.count))
    return vote

def acc_seq(y_pred,Y):
    res=[]
    for i in range(0,len(Y)):
        temp=[]
        for j in range(0,60):
            temp.append(y_pred[i*60+j])
        res.append(temp)
    vote_test=maj_vote(res)
    print(accuracy_score(vote_test,np.argmax(Y, axis=1)))


experiment_name = 'Classify_mvts'
train = ds.DataSet('../data2/',720, onehot=False, load=True)

#print(train.label.shape)
#print(train.data.shape)


X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train.data, train.label, train_size=0.75, test_size=0.25)
x_train,x_test,y_train,y_test=[],[],[],[]


for i in X_train:
    for j in i:
        x_train.append(j)
for i in X_test:
    for j in i:
        x_test.append(j)
for i in Y_train:
    for j in range(0,60):
        y_train.append(i)
for i in Y_test:
    for j in range(0,60):
        y_test.append(i)

print(np.array(y_train).shape)
rf=RandomForestClassifier(n_estimators=50,n_jobs=-1,criterion="gini")
rf.fit(x_train,y_train)
with open("weights/rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)
print("trained model!")
#print(rf.score(x_train,y_train))

y_pred_train=rf.predict(x_train)
y_pred_test=rf.predict(x_test)
#print(confusion_matrix(y_train,y_pred))

Y_test=[]
for i in range(0,len(y_test),60):
    Y_test.append(y_test[i])

time_n = time.time()
acc_seq(y_pred_test,Y_test)
print("time between frames:", (time.time()-time_n)/240)
