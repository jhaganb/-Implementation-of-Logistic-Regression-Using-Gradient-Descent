# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and Load the dataset.
2. Define X and Y array and Define a function for costFunction,cost and gradient.
3. Define a function to plot the decision boundary.
4. Define a function to predict the Regression valu

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Jhagan B
RegisterNumber:  212220040066
*/

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```

## Output:
Array value of x:

![image](https://github.com/jhaganb/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/63654882/199fc4ac-5efc-462a-8772-6a793104031e)

Array value of y:

![image](https://github.com/jhaganb/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/63654882/43b5a59c-3057-4328-bfed-7a0d51fa804d)

Score graph:

![image](https://github.com/jhaganb/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/63654882/6910e0b6-1792-442b-88b4-8f17f7c189f1)

Sigmoid function graph:

![image](https://github.com/jhaganb/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/63654882/ef2323b0-df55-4f91-bc22-0d810d5549ae)

X train grad value:

![image](https://github.com/jhaganb/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/63654882/48e70838-de4f-4f00-be7a-ce45ea60c7f3)

Y train grad value:

![image](https://github.com/jhaganb/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/63654882/ae1ed0df-1d2b-43df-a129-98368a48386a)

Regression value:

![image](https://github.com/jhaganb/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/63654882/5b8d4a44-0b2f-4dad-a52c-65f021c6bd92)

decision boundary graph:

![image](https://github.com/jhaganb/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/63654882/232a9617-72c5-4e2b-8424-f0e23efe8376)

Probability value:

![image](https://github.com/jhaganb/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/63654882/20425408-7d55-4f07-a6cd-38058e0c5655)

Prediction value of mean:

![image](https://github.com/jhaganb/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/63654882/29823a97-e9de-47f9-997f-2f76a335e4e2)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

