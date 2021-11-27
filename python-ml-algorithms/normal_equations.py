import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Regularization uygulanmıştır.
def normalEquation(X,Y,lamda): # (X.T * X + lamda * identifity(n+1,n+1))^-1 * X.T * Y
    t = np.eye(X.shape[1])
    t[0][0] = 0
    pinv = np.linalg.inv(np.dot(X.T,X) + lamda * t)
    W = np.dot((np.dot(pinv,X.T)),Y)
    return W
    
def prediction(X,W):
    predict = np.dot(X,W)
    return predict

data = pd.read_csv('house_price_dataset.txt', sep=",", header = None)
data = data.to_numpy()
X = data[:,0:2]
Y = data[:,2:]
X = np.append(np.ones((Y.shape[0],1)), X ,axis=1)
W = normalEquation(X,Y,lamda=100)

pre = prediction(np.array([[1,1650,3]]),W)
print(pre)

c = np.arange(0,Y.shape[0]).reshape((Y.shape[0],1))
f=plt.figure(figsize=(12,6))
plt.scatter(x=c,y=Y,s=75,marker="X",)
plt.plot((np.dot(X,W)), color="r")
plt.xlabel("Örnekler")
plt.ylabel("Sonuçlar")
plt.title("Ev Fiyat Tahmini")
plt.legend()
plt.show()