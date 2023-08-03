import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

hello=pd.read_csv("creditcard2.csv")
print("hello=")
print(hello)
y=hello[['Class']].to_numpy()
y_train=y[:227845][:]
y_train=y_train.T
print("y_train=")
print(y_train)
y_test=y[227845:][:]
y_test=y_test.T
print("y_test=")
print(y_test)

x=hello.to_numpy()
x=np.delete(x,0,1)
x=np.delete(x,29,1)

scaler = StandardScaler()
x=scaler.fit_transform(x)
print("x=")
print(x)

x_train=x[:227845][:]
x_train=x_train.T
print("x_train=")
print(x_train)
x_test=x[227845:][:]
x_test=x_test.T
print("x_test=")
print(x_test)



def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s

def initialization(n_x):
    w = np.random.randn(n_x,1)*0.01
    b = 0.0
    return w,b


def propagate(w, b, x, y):
    
    Z=np.dot(w.T,x)+b

    A=sigmoid(Z)
    
    return A

def gradient_descent(w,b,x,y,n_i,lr):
    
    m=x.shape[1]
    
    for i in range(n_i):
        A=propagate(w,b,x,y)
      

        dw = (np.dot(x,(A-y).T))/m
      
        db = (np.sum(A-y))/m
        
        w = w - (lr*dw)
        b = b - (lr*db)
       
        
    return w,b

def prediction(w,b,x):
    m = x.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(x.shape[0], 1)
    
    A = sigmoid(w.T.dot(x) + b)
    
    for i in range(A.shape[1]):
        if A[0, i] <= 0.5 :
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
    
    return Y_prediction

def model(x_train, y_train, x_test, y_test, n_i, lr):
   
    w,b=initialization(x_train.shape[0])
    
    w,b=gradient_descent(w, b, x_train, y_train, n_i, lr)
    
    
    Y_prediction_test = prediction(w, b, x_test)
    Y_prediction_train = prediction(w, b, x_train)
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - y_test)) * 100))
    
model(x_train, y_train, x_test, y_test, n_i = 2000, lr = 0.005)