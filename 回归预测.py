# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 18:51:35 2022

@author: jiayiliu
"""

#多元回归，想要预测总费用
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#利用sklearn里面的包来对数据集进行划分，以此来创建训练集和测试集
#train_size表示训练集所占总数据集的比例
loan_data=pd.read_excel(r'C:\Users\张睿\Desktop\dnn_house_price_prediction_scratch-master\data\loan.xlsx', header=0)
data=loan_data.iloc[:,:-1]
label=loan_data.iloc[:,-1]
X_train,X_test,Y_train,Y_test = train_test_split(data,label,train_size=.80)
print("原始数据特征:",data.shape,
      ",训练数据特征:",X_train.shape,
      ",测试数据特征:",X_test.shape)
 
print("原始数据标签:",label.shape,
      ",训练数据标签:",Y_train.shape,
      ",测试数据标签:",Y_test.shape)
model = LinearRegression()
model.fit(X_train,Y_train)
a  = model.intercept_#截距
b = model.coef_#回归系数
print("最佳拟合线:截距",a,",回归系数：",b)
score = model.score(X_test,Y_test)
print("训练的回归模型的在测试集上验证的得分是：")
print(score)
#对线性回归进行预测
Y_pred = model.predict(X_test)
print("训练的回归预测模型，预测的训练集的结果是：")
print(Y_pred)
plt.plot(range(len(Y_pred)),Y_pred,'y',label="predict")
#显示图像
# plt.savefig("predict.jpg")
plt.show()
plt.figure()
plt.plot(range(len(Y_pred)),Y_pred,'y',label="predict")
plt.plot(range(len(Y_pred)),Y_test,'g',label="test")
plt.legend(loc="upper right") #显示图中的标签
plt.xlabel("the number of sales")
plt.ylabel('value of sales')
plt.show()