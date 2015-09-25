# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:52:35 2015

@author: user
"""
import arff,reduction,numpy as np,matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def showReduction(bunch,reduction,name='dataset',lege=True):
    X = bunch.data
    y = bunch.target
    target_names = bunch.target_names

    X_r,reductionName=reduction(X,2)
    N=len(target_names)
    
    numbers=range(N)
  
    plt.figure()
    pointShape='ovs'
    for i, target_name in zip( numbers, target_names):
        m=pointShape[i / 9]
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1],c=getColor(i),marker=m, label=target_name)
    if(lege):
        plt.legend()
    plt.title(reductionName+' of '+name)
    plt.show()

def showReductionAdnotated(bunch,reduction,name='dataset',lege=True):  
    x = bunch.data
    y = bunch.target
    target_names = bunch.target_names
    X_r,reductionName=reduction(x,2)
    X_r=np.transpose(X_r)
    fig, ax = plt.subplots()
    numbers=range(len(target_names))
    #c_p=[getColor(i) for i in target_names ]
    #ax.scatter(X_r[0], X_r[1],c=c_p)
    pointShape='ovs'
    for i, target_name in zip( numbers, target_names):
        m=pointShape[i / 9]
        plt.scatter(X_r[0,y == i], X_r[1,y == i],c=getColor(i),marker=m, label=target_name)
    for i,txt in enumerate(list(y)):
        print(type(X_r[0,i])) 
        ax.annotate(str(txt),(X_r[0,i], X_r[1,i]))
    plt.title(reductionName+' of '+name)
    plt.show()

def showReduction3D(bunch,reduction,name='dataset'):
    X = bunch.data
    y = bunch.target
    target_names = bunch.target_names

    X_r,reductionName=reduction(X,3)
    N=len(target_names)
    numbers=range(N)
  
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pointShape='ovs'
    for i, target_name in zip( numbers, target_names):
        m=pointShape[i / 9]
        color=getColor(i)
        ax.scatter(X_r[y == i, 0], X_r[y == i, 1],X_r[y == i,2],c=color,marker=m, label=target_name)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title(reductionName +' of ' + name)
    plt.show()

def getArea(index):
    div=index / 9
    return np.pi * (3+div)**2
    
def getColor(index):
    print("qwerty")
    print(type(index))
    cls="bgrcmykw"
    i=index % len(cls)
    return cls[i]

path="/home/user/Desktop/DeepActionLearning/"
name= path+"raw.arff"   
dataset=arff.readArffDataset(name)  
showReductionAdnotated(dataset,reduction.tsneReduction,"3_12_8",False)
#reduction.reduceDataset(name,200,reduction.pcaReduction)
       

#dataset=arff.readArffDataset("C:/Users/user/Desktop/kwolek/output/3_8_4_0.arff")
#showReduction3D(dataset,reduction.mdaReduction)
#show2D(pcaReduction(dataset))
