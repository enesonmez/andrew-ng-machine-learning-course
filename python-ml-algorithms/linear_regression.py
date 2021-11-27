import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Sütunlar : özellik (featue)
# Satırlar : örnek

# L2 regularization kullanılmıştır.
def costFunction(X,b,Y,W,lamda): # J = (1/2m) * (∑ (h - Y)^2 + lamda * ∑ W^2)
    m = Y.shape[0]
    A = (np.dot(X,W) + b) - Y
    J = (1 / (2*m)) * (np.dot(A.T,A) + lamda * np.dot(W.T,W))
    return J

def graidentDescent(X,Y,W,b,lamda,learningrate,epochs): # W = W - learningrate * [(1/m) * ∑ (h - Y)*X + (lamda/m) * W)
    m = Y.shape[0]                                      # b = b - learningrate * (1/m) * ∑ (h - Y)
    J_history = []

    for _ in range(epochs):
        A = (np.dot(X,W) + b) - Y
        dW = (1/m) * np.dot(X.T,A) + (lamda / m) * W
        db = (1/m) * np.sum(A,axis=0,keepdims=True)
        W = W - learningrate * dW
        b = b - learningrate * db
        
        J_history.append(costFunction(X,b,Y,W,lamda))
    return W, b, J_history

def prediction(X,W,b): # Tahmin işlemi için kullanılır.
    predict = np.dot(X,W) + b
    return predict

def meanNormalization(X): # Feature'ları normalize etmek için kullanılır.
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    X = (X - mu) / sigma
    return mu, sigma, X


""" Veri seti dahil etme ve model eğitimi"""
data = pd.read_csv('house_price_dataset.txt', sep=",", header = None)
data = data.to_numpy()
X = data[:,0:2]
Y = data[:,2:]
b = np.zeros((1,1))
W = np.zeros((X.shape[1],1))

mu, sigma, X = meanNormalization(X)
W, b , j = graidentDescent(X,Y,W,b,lamda=1,learningrate=0.3,epochs=50)
pre = (np.array([[1650,3]]) - mu) / sigma
print((prediction(pre,W,b)))

c = np.arange(0,Y.shape[0]).reshape((Y.shape[0],1))
f=plt.figure(figsize=(12,6))
plt.scatter(x=c,y=Y,s=75,marker="X",)
plt.plot((np.dot(X,W)+b), color="r")
plt.xlabel("Örnekler")
plt.ylabel("Sonuçlar")
plt.title("Ev Fiyat Tahmini")
plt.legend()
plt.show()