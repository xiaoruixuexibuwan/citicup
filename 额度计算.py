# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 21:29:10 2022

@author: 张睿
"""
import pickle 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold 
import numpy as np


with open(r'C:\Users\张睿\Desktop\loan_prediction\loan.obj', 'rb') as f: 
    clf = pickle.load(f) 

loan_test=pd.read_excel(r'C:\Users\张睿\Desktop\test_data.xlsx', header=0)
labels=loan_test.iloc[:,:-1]
preds = clf.predict_proba (labels) 
predict = pd.DataFrame(preds[:,-1])
predict.columns=["loan_predict"]
print(predict.describe())
predict.to_excel(r'C:\Users\张睿\Desktop\text_sample.xlsx')
predict.mode()

def loan_amount (predict,F,B,p,H):
    '''
    

    Parameters
    ----------
    predict : dataframe
        申请用户的违约违约概率
    F : float64
        最低额度
    B : float64
        中间额度
    p : float64
        新用户申请贷款违约概率
    H : float64
        最高额度.

    Returns
    -------
    f : float64
        贷款额度.

    '''
    if p > predict.mode():
        f = 1+(F/B-1)/(predict.max()-predict.mode())*(p-predict.mode())
    elif p < predict.mode():
        f = 1+(H/B-1)/(predict.min()-predict.mode())*(p-predict.mode())
    else:
        f=B
    return f

def loan_amount (predict,Rmax,R,p,Rmin):
    '''
    

    Parameters
    ----------
    predict : DataFrame
        申请用户的违约违约概率
    Rmax : float64
        最低额度利率.
    R : float64
        中间额度利率.
    p : float64
        新用户申请贷款违约概率.
    Rmin : float64
        最高额度利率

    Returns
    -------
    r : float64
        贷款利率.

    '''

    if p > predict.mode():
        r = 1+(Rmax/R-1)/(predict.max()-predict.mode())*(p-predict.mode())
    elif p < predict.mode():
        r = 1+(Rmin/R-1)/(predict.min()-predict.mode())*(p-predict.mode())
    else:
        r=R
    return r





