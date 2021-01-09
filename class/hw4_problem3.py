#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample code of HW4, Problem 3
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np
import math
from scipy import linalg

myfile = open('hw4_p3_data.pickle', 'rb')

mydict = pickle.load(myfile)

X_train = mydict['X_train']
X_test = mydict['X_test']
Y_train = mydict['Y_train']
Y_test = mydict['Y_test']

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

predictive_mean = np.empty(X_test.shape[0])
predictive_std = np.empty(X_test.shape[0])



sigma = 0.1
sigma_f = 1.0
ls = 0.06

#-------- Your code (~10 lines) ---------
def sigmaij(xi, xj):
    if xi==xj:
        delta=1
    else:
        delta=0
    answer=sigma_f*sigma_f*math.exp((xi-xj)*(xi-xj)/(2*ls*ls))+sigma*sigma*delta
    return answer
def xkx1n (xk):
    answer=np.empty(X_train)
    for i in X_train:
        answer[i]=sigmaij(xk,i)
    return answer

kx1nx1n=np.empty(X_train[0].shape,X_train[0])
for i in range(0,X_train.shape[0]):
    for j in range(0,X_train.shape[0]):
        kx1nx1n[i][j]=sigma(X_train[i],X_train[j])


I=np.identity(X_test.shape[0])

y1n=np.shape(Y_train,1)

for i in predictive_mean:
    print(i)


    
#---------- End of your code -----------

# Optional: Visualize the training data, testing data, and predictive distributions
fig = plt.figure()
plt.plot(X_train, Y_train, linestyle='', color='b', markersize=3, marker='+',label="Training data")
plt.plot(X_test, Y_test, linestyle='', color='orange', markersize=2, marker='^',label="Testing data")
plt.plot(X_test, predictive_mean, linestyle=':', color='green')
plt.fill_between(X_test.flatten(), predictive_mean - predictive_std, predictive_mean + predictive_std, color='green', alpha=0.13)
plt.fill_between(X_test.flatten(), predictive_mean - 2*predictive_std, predictive_mean + 2*predictive_std, color='green', alpha=0.07)
plt.fill_between(X_test.flatten(), predictive_mean - 3*predictive_std, predictive_mean + 3*predictive_std, color='green', alpha=0.04)
plt.xlabel("X")
plt.ylabel("Y")
