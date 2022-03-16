# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 18:52:04 2022

@author: jiayiliu
"""

# 分层交叉验证
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold #分层k折交叉验证
import numpy as np
kf = StratifiedKFold(n_splits=10, shuffle=True)
csv_4=csv
datasets=csv_4.iloc[:,1:11]
labels=csv_4.iloc[:,12]
num=0
mean_score_f1=0
mean_score_recall=0
mean_score_precision=0
for train_index, test_index in kf.split(csv_4.iloc[:,1:11], csv_4.iloc[:,12]):
    x_train = csv_4.iloc[train_index,1:11]
    y_train = csv_4.iloc[train_index,12]
    x_test = csv_4.iloc[test_index,1:11]
    y_test = csv_4.iloc[test_index,12]
    clf = RandomForestClassifier(n_estimators=200)
    clf.fit(x_train, y_train)
    num=num+1
    print("第",num,'次')
    preds = clf.predict(x_test)
    precision = round(metrics.precision_score(y_test, preds,average='macro') * 100, 2)
    
    print('precision_score:',precision)
    recall=round(metrics.recall_score(y_test, preds,average='macro') * 100, 2)
    print("recall_score:",recall)
    f1=round(metrics.f1_score(y_test, preds,average='macro') * 100, 2)
    print("f1_score:",f1)
    print(pd.crosstab(y_test, preds, rownames=['actual'], colnames=['preds']))
    print('*'*50)
    mean_score_precision=mean_score_precision+precision
    mean_score_f1=mean_score_f1+f1
    mean_score_recall=mean_score_recall+recall
    dict_1={}
    values=clf.feature_importances_
    #print(len(values))
    #print(values)
    

print('十交叉验证 precision 平均得分是：',(mean_score_precision/10))
print('十交叉验证 recall 平均得分是：',(mean_score_recall/10))
print('十交叉验证 f1 平均得分是：',(mean_score_f1/10))