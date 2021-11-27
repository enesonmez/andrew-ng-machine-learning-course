import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Sütunlar : özellik (featue)
# Satırlar : örnek

#Aktivasyon Fonksiyonu
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

#İleri ve geri yayılım fonksiyonu
def propagation(X,Y,W,b,lamda):
    m = X.shape[0]
    Z = np.dot(X,W) + b
    A = sigmoid(Z)
    #Forward Propagation (Cost Function)    L2 regularization uygulanmıştır.
    J = (1 / m) * (np.dot(np.log(A.T),-Y) - np.dot(np.log((1-A).T),(1 - Y))) + lamda/(2*m)*np.dot(W.T,W)
    
    #Backward Propagation
    dz = A - Y
    dw = (1/m) * np.dot(X.T,dz) + (lamda/m) * W
    db = (1/m) * np.sum(dz,axis=0)
    grad = {"dw":dw, "db":db}

    return grad, J

def gradientDescent(X,Y,W,b,epochs,learning_rate,lamda,print_cost=False):
    J_history = []
    for i in range(epochs):
        grad, J = propagation(X,Y,W,b,lamda)
        db = grad["db"]
        dw = grad["dw"]

        W = W - learning_rate * dw
        b = b - learning_rate * db

        if i % 50 == 0:
            J_history.append(J)
        if print_cost and i % 50 == 0:
            print("Cost after iteration {}: {}".format(i,J))
    parameter = {"w":W,"b":b}
    return parameter, J_history

def meanNormalization(X): # Feature'ları normalize etmek için kullanılır.
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    X = (X - mu) / sigma
    return mu, sigma, X

def prediction(X,W,b): # Tahmin işlemi için kullanılır.
    predict = sigmoid(np.dot(X,W) + b)
    predict = (predict>0.5)
    return predict

def accuracy(predict,Y): # Başarı oranı kontrol edilir.
    accury = np.squeeze(((np.dot(Y.T,predict) + np.dot((1-Y).T,1-predict))/float(Y.size)) * 100)
    return accury


def plotData(X,Y):
    plt.figure(figsize=(12,6))
    pos = np.where(Y==1)
    neg = np.where(Y==0)
    plt.scatter(x=X[pos,0],y=X[pos,1],s=75,marker="X",label="geçti",color="green")
    plt.scatter(x=X[neg,0],y=X[neg,1],s=75,marker="o",label="kaldı",color="red")
    plt.legend()
    plt.show()  

""" Veri seti dahil etme ve model eğitimi"""
data = pd.read_csv('exam_score_dataset.txt', sep=",", header = None)
data = data.to_numpy()
X = data[:,0:2]
Y = data[:,2:]
b = np.zeros((1,1))
W = np.zeros((X.shape[1],1))
plotData(X,Y)
"""mu, sigma, X = meanNormalization(X)
#Model eğitimi
parameter, J = gradientDescent(X,Y,W,b,epochs=1001,learning_rate=0.3,lamda=10,print_cost=True)
print(parameter)
#Tahmin ve başarı oranı tespiti
p = prediction(X,parameter["w"],parameter["b"])
print(accuracy(p,Y))
pre = (np.array([[55,90]]) - mu) / sigma
print(prediction(pre,parameter["w"],parameter["b"]))"""