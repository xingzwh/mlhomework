import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
data=np.loadtxt("D:\workspace\上课\机器学习\insurance.csv",delimiter=',',dtype=str,skiprows=1)
print(data)
data[data == 'male'] = 1
data[data =='female'] = 0
data[data == 'yes'] = 1
data[data == 'no'] = 0
'''
southeast=010
southwest=100
northeast=000
northwest=001
'''
print(data)
X=data[:,:8]
print(X)
Y=data[:,8]
print(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
model=linear_model.LinearRegression();
model.fit(X_train,Y_train)
print(model.coef_)
Y_train_pred = model.predict(X_train)
Y_predict=model.predict(X_test)
'''print(Y_predict)
print(Y_predict.size)
print(Y_test)
print(Y_test.size)'''
float_Y_train = Y_train.astype(np.float32)
float_Y_train_pred = Y_train_pred.astype(np.float32)
float_Y_test = Y_test.astype(np.float32)
float_Y_predict = Y_predict.astype(np.float32)
print(float_Y_test)
print(float_Y_predict)

print("MSE of train: %.2f, test, %.2f" % (
    mean_squared_error(float_Y_train,float_Y_train_pred),
    mean_squared_error(float_Y_test,float_Y_predict)))

print("R^2 of train: %.2f, test, %.2f" % (
    r2_score(float_Y_train,float_Y_train_pred),
    r2_score(float_Y_test,float_Y_predict)))