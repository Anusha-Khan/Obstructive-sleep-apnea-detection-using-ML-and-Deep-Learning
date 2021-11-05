import numpy as nmpy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import  RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import metrics

import argparse, os
parser = argparse.ArgumentParser(description="ML-OSA")
parser.add_argument("--train_input", default= "/content/drive/MyDrive/apnea-ecg-database-1.0.0/train_input.npy", type=str, help="train_input")
parser.add_argument("--train_label", default= "/content/drive/MyDrive/apnea-ecg-database-1.0.0/train_label.npy", type=str, help="train_label")
global opt, model
opt = parser.parse_args()

#train_input = opt.train_input
#train_label = opt.train_label
train_input = nmpy.load(opt.train_input, allow_pickle=True)
train_label = nmpy.load(opt.train_label, allow_pickle=True)


def split_data(train_input):
    X1 = []
    X2 = []
    for index in range(len(train_input)):
        X1.append([train_input[index][0], train_input[index][1]])
        X2.append([train_input[index][2], train_input[index][3]])

    return nmpy.array(X1).astype('float64'), nmpy.array(X2).astype('float64')



def get_data(train_input,train_label):

    X_train, X_test, y_train, y_test = train_test_split(train_input, train_label, test_size=0.20, random_state=42)
    X_train1, X_train2 = split_data(X_train)
    X_test1,  X_test2   = split_data(X_test)

    X_train1 = nmpy.transpose(X_train1, (0, 2, 1))
    X_train2 = nmpy.reshape(X_train2, (X_train2.shape[0], X_train2.shape[1], 1))
    X_test1  = nmpy.transpose(X_test1, (0, 2, 1))
    X_test2  = nmpy.reshape(X_test2, (X_test2.shape[0], X_test2.shape[1], 1))

    return X_train1, X_train2, y_train, X_test1, X_test2, y_test

# call split data  function 
X_train1, X_train2, y_train, X_test1, X_test2, y_test = get_data(train_input,train_label)

#values present are on different scale hence we need to standardize them so 
#that our gradient descent can reach the minima at a faster pace
def scale(train_input):
    scalers = {}
    for i in range(train_input.shape[2]):
        scalers[i] = StandardScaler()
        train_input[:, i, :] = scalers[i].fit_transform(train_input[:, i, :]) 
    return train_input

#scale the train and test data 

X_train1 = scale(X_train1)
X_train2 = scale(X_train2)


X_test1 = scale(X_test1) 
X_test2 = scale(X_test2)

X_train1.shape

nsamples, nx, ny = X_train1.shape
#change data to 3d ndarray to 2d array to give it to ML model
d2_train_dataset = X_train1.reshape((nsamples,nx*ny))

nsamples, nx, ny = X_test1.shape
#change data to 3d ndarray to 2d array to give it to ML model
d2_test_dataset = X_test1.reshape((nsamples, nx*ny))

"""## ML Classification Algorithms

### Support vector machine
"""

svc = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale',random_state=0)



svc_fit = svc.fit(d2_train_dataset,y_train)
y_pred_svc = svc_fit.predict(d2_test_dataset)
confusionMatrixSVC = pd.crosstab(y_test,y_pred_svc, rownames=['Actual'], colnames=['Predicted'], margins=True)
print(confusionMatrixSVC)

AccuracySVM = metrics.accuracy_score(y_test, y_pred_svc)
print ('Accuracy',AccuracySVM)

"""### Random Forest Classifier"""

rfc = RandomForestClassifier()


rfc_fit=rfc.fit(d2_train_dataset,y_train)
y_pred_RFC = rfc_fit.predict(d2_test_dataset)

confusionMatrixSVC = pd.crosstab(y_test,y_pred_RFC, rownames=['Actual'], colnames=['Predicted'], margins=True)
print(confusionMatrixSVC)

AccuracyRFC = metrics.accuracy_score(y_test, y_pred_RFC)
print ('Accuracy',AccuracyRFC)

"""### Decesion Trees"""

#decesion trees

dtc = DecisionTreeClassifier()
dtc_fit = dtc.fit(d2_train_dataset,y_train)
dtc.score(d2_train_dataset,y_train)
y_pred_dt = svc_fit.predict(d2_test_dataset)
print(dtc.score(d2_test_dataset,y_pred_dt))
confusionMatrixDTC = pd.crosstab(y_test,y_pred_dt, rownames=['Actual'], colnames=['Predicted'], margins=True)
print(confusionMatrixDTC)


AccuracyDTC = metrics.accuracy_score(y_test, y_pred_dt)
print ('Accuracy',AccuracyDTC)

"""### K Nearest Neighbour"""

#KNN

knn = KNeighborsClassifier()
knn_fit = knn.fit(d2_train_dataset,y_train)
knn.score(d2_train_dataset,y_train)
y_pred_KNN = knn_fit.predict(d2_test_dataset)

confusionMatrixKNN = pd.crosstab(y_test,y_pred_KNN, rownames=['Actual'], colnames=['Predicted'], margins=True)
print(confusionMatrixKNN)

AccuracyKNN = metrics.accuracy_score(y_test, y_pred_KNN)
print ('Accuracy',AccuracyKNN)

"""### Adaptive Boosting Algorithm"""

#AdaBoost

ADA_fit = AdaBoostClassifier(random_state=0).fit(d2_train_dataset,y_train)
y_pred_ADA = ADA_fit.predict(d2_test_dataset)
confusionMatrixADA = pd.crosstab(y_test,y_pred_ADA, rownames=['Actual'], colnames=['Predicted'], margins=True)
print(confusionMatrixADA)


AccuracyADA = metrics.accuracy_score(y_test, y_pred_ADA)
print ('Accuracy',AccuracyADA)

"""### Gaussian Naive bayes Classifoer"""

GNB_fit = GaussianNB().fit(d2_train_dataset,y_train)
y_pred_GNB = GNB_fit.predict(d2_test_dataset)
confusionMatrixGNB = pd.crosstab(y_test,y_pred_GNB, rownames=['Actual'], colnames=['Predicted'], margins=True)
print(confusionMatrixGNB)

AccuracyGNB = metrics.accuracy_score(y_test, y_pred_GNB)
print ('Accuracy',AccuracyGNB)

"""### Quadratic Discriminant Analysis"""

QDA_fit = QuadraticDiscriminantAnalysis().fit(d2_train_dataset,y_train)
y_pred_QDA = QDA_fit.predict(d2_test_dataset)
confusionMatrixQDA = pd.crosstab(y_test,y_pred_QDA, rownames=['Actual'], colnames=['Predicted'], margins=True)
print(confusionMatrixQDA)


AccuracyQDA = metrics.accuracy_score(y_test, y_pred_QDA)
print ('Accuracy',AccuracyQDA)

"""## Comparision OF accuracy of ML Algorithms for OSA detectiion."""

import matplotlib.pyplot as plt

results = {u'SVM':AccuracySVM, u'RFC': AccuracyRFC, u'DTC':AccuracyDTC, u'KNN':AccuracyKNN, u'ADAbosst':AccuracyADA
           , u'GNB':AccuracyGNB, u'QDA':AccuracyQDA}

import json


json = json.dumps(results)
f = open("MLresults.json","w")
f.write(json)
f.close()

plt.bar(*zip(*results.items()))
plt.show()