import pandas as pd
import numpy as np
from pandas import DataFrame,Series
#import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
#%matplotlib inline
np.set_printoptions(precision=2)
data = pd.read_csv('C:/Users/12733/Desktop/AI_lab3/car.data.txt',encoding = 'utf-8',header = None)
data.columns.tolist()
data.rename(columns = {0:'buying',1:'maintainence',2:'doors',3:'persons',4:'lug_boot',5:'safety',6:'class'},inplace = True)
data['class'].value_counts()
cleanup_nums = {"class":     {"unacc": 4, "acc": 3,'good': 2,'vgood':1}
                }
data.replace(cleanup_nums,inplace = True)
data['class'].value_counts()
data.dtypes
target = data['class']
data.drop( ['class'],axis = 1,inplace = True)
target.head(1)
data = pd.get_dummies(data)
data.head()
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X_train,X_test,y_train,y_test = train_test_split(data,target,random_state = 0)
from sklearn.svm import LinearSVC
for this_c in [1, 3, 5,7,9,11,13]:
    clf = LinearSVC(C = this_c).fit(X_train, y_train)
# print('Accuracy of Linear SVC classifier on training set: {:.2f}'
#      .format(clf.score(X_train, y_train)))
# print('Accuracy of Linear SVC classifier on test set: {:.2f}'
#      .format(clf.score(X_test, y_test)))
    r2_train = clf.score(X_train, y_train)
    r2_test = clf.score(X_test, y_test)
    print('Cs = {:.2f}, \r-squared training: {:.2f}, r-squared test: {:.2f}\n'.format(this_c, r2_train, r2_test))
from sklearn.svm import SVC
print('SVC RBF: effect of  regularization parameter gamma\n')
for this_gamma in [0.01, 1, 10,50]:
    clf = SVC( kernel = 'rbf', gamma=this_gamma).fit(X_train, y_train)
    r2_train = clf.score(X_train, y_train)
    r2_test = clf.score(X_test, y_test)
print('Gamma = {:.2f}, \r-squared training: {:.2f}, r-squared test: {:.2f}\n'.format(this_gamma, r2_train, r2_test))
print('SVC RBF: effect of  regularization parameter gamma and C \n')
for this_gamma in [1, 10,50,100]:
    for this_C in [0.1, 1, 15]:
        
        clf = SVC(    kernel = 'rbf', gamma=this_gamma, C = this_C).fit(X_train, y_train)
        r2_train = clf.score(X_train, y_train)
        r2_test = clf.score(X_test, y_test)
        print('Gamma = {:.2f},C = {:.2f} \r-squared training: {:.2f}, r-squared test: {:.2f}\n'.format(this_gamma,this_C, r2_train, r2_test))